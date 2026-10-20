"""Notebook-level orchestration of M18 (T15–T17). Upstream runs are READ only; the primary family is pre-registered before
the first outcome-bearing step."""

from __future__ import annotations

from typing import Any

from ..status import BLOCKED_MISSING_INPUT, ResearchBlocked
from . import evidence as EVD
from . import ladder as LD
from . import position as PS
from . import reporting as RP
from . import robustness as RB
from . import scouting as SC
from . import specificity as SP


def target_columns(cfg: Any) -> dict[str, str]:
    """{target_id: column}: primary + (when configured) the M16-frozen eligible task targets."""
    import json
    from pathlib import Path

    from ..m16.targets import TARGETS

    out = {cfg.primary_target: TARGETS[cfg.primary_target].column}
    if cfg.secondary_targets == "eligible":
        p = Path(cfg.m16_run_dir) / "manifests" / "target_freeze_m16.json"
        if p.is_file():
            for t in json.loads(p.read_text(encoding="utf-8")).get("eligible_task_targets", []):
                out[t] = TARGETS[t].column
    return out


def prepare(cfg: Any, exp: Any, run: Any, ctx: dict[str, Any]) -> dict[str, Any]:
    """Analysis frame (+ targets), rung eligibility and the immutable pre-registration."""
    import polars as pl

    from ..m16.baselines import m14_reliable_table
    from ..m17.signatures import validated_columns

    data = LD.analysis_with_signatures(cfg)
    tcols = target_columns(cfg)
    from pathlib import Path

    tg = Path(cfg.m16_run_dir) / "features" / "targets" / "athlete_targets.parquet"
    if tg.is_file():
        t = pl.read_parquet(tg).with_columns(pl.col("nfl_id").cast(pl.Int64))
        add = [c for c in t.columns if c not in data.columns and c != "nfl_id"]
        data = data.join(t.select("nfl_id", *add), on="nfl_id", how="left")
    try:
        _m14, m14_cols, _note = m14_reliable_table(cfg.m14_run_dir)
    except ResearchBlocked:
        m14_cols = []
    gates = LD.gate_tables(cfg)
    sig_cols = validated_columns(gates["g17"]) if gates.get("g17") is not None else []
    rungs, frame = LD.eligible_rungs(cfg, gates, data, [c for c in m14_cols if c in data.columns], [c for c in sig_cols if c in data.columns])
    fam = [LD.NAMES[r] for r in ("B2", "B3", "B4", "B5", "B6", "B7") if rungs[r]["status"] == "eligible"]
    prereg = run.preregister({"primary_target": cfg.primary_target, "primary_family": fam, "reference": "B1_M1", "targets": tcols,
                              "rungs": {k: {kk: str(vv) for kk, vv in v.items()} for k, v in rungs.items()}})
    ctx["outputs"] += run.table(frame, "rung_eligibility", "ladder")
    return {"data": data, "targets": tcols, "rungs": rungs, "rung_frame": frame, "prereg": prereg}


def run_a(cfg: Any, exp: Any, P: dict[str, Any], run: Any, ctx: dict[str, Any]) -> dict[str, Any]:
    import polars as pl

    fold_of = exp.folds_by_athlete
    log: list = []
    mods, blocks = LD.build_models(cfg, P["rungs"], P["data"], fold_of, log)
    res = {}
    for tid, col in P["targets"].items():
        if col not in P["data"].columns:
            continue
        raw, ladder_log = LD.run_ladder(exp, P["data"], col, mods)
        ev = LD.evaluate(raw, fold_of, tid, tid == cfg.primary_target, cfg)
        ev["raw"] = raw
        res[tid] = ev
        run.data_file(raw, f"predictions/oof_{tid}", {"target": tid, "models": list(mods)})
    if cfg.primary_target not in res:
        raise ResearchBlocked(BLOCKED_MISSING_INPUT, f"primary target {cfg.primary_target} not in the analysis frame", missing=[cfg.primary_target])
    for k in ("metrics", "paired", "status", "folds", "intervals"):
        fs = [r[k] for r in res.values() if r.get(k) is not None and r[k].height]
        if fs:
            ctx["outputs"] += run.table(pl.concat(fs, how="diagonal_relaxed"), k, "ladder")
    run.data_file(pl.concat([r["long"] for r in res.values()], how="diagonal_relaxed"), "predictions/oof_long", {"targets": list(res)})
    best = None
    st = res[cfg.primary_target]["paired"]
    if st is not None and st.height:
        cand = st.filter(pl.col("model").str.starts_with("B") & (pl.col("reference") == "M1")).sort("delta_rmse", descending=True)
        best = cand["model"][0] if cand.height else None
    return {"mods": mods, "blocks": blocks, "results": res, "best": best, "primary": res[cfg.primary_target]}


def run_b(cfg: Any, exp: Any, P: dict[str, Any], A: dict[str, Any], run: Any, ctx: dict[str, Any]) -> dict[str, Any]:
    from ..m14.transfer import by_position

    prim = A["primary"]
    col = P["targets"][cfg.primary_target]
    tab = PS.run_positions(exp, P["data"], col, A["mods"], prim["raw"], prim["strict_ids"], cfg.min_position_n, max(200, cfg.n_boot // 5), cfg.seed)
    verdict = PS.position_verdict(tab)
    _m, bp = by_position(prim["wide"], prim["models"], cfg.min_position_n, max(200, cfg.n_boot // 5), cfg.seed)
    for name, df in (("separate_vs_pooled", tab), ("position_verdict", verdict), ("pooled_by_position", bp)):
        ctx["outputs"] += run.table(df, name, "position")
    return {"table": tab, "verdict": verdict, "by_position": bp}


def run_c(cfg: Any, P: dict[str, Any], run: Any, ctx: dict[str, Any]) -> dict[str, Any]:
    from ...data.validation import M1_COLS
    from ..m16.targets import TARGETS

    tc = {t: TARGETS[t].column for t in ("T1", "T2", "T3", "T4", "T5") if TARGETS[t].column in P["data"].columns}
    feats = [c for c in P["data"].columns if SP.family_of(c) is not None]
    if not tc or not feats:
        raise ResearchBlocked(BLOCKED_MISSING_INPUT, "task targets or drill features missing for the specificity matrix", missing=["t16_* / trk_* columns"])
    long, contrast = SP.matrix(P["data"], tc, feats, M1_COLS, max(200, cfg.n_boot // 5), cfg.seed)
    ctx["outputs"] += run.table(long, "specificity_matrix", "specificity") + run.table(contrast, "diagonal_contrast", "specificity")
    return {"long": long, "contrast": contrast}


def run_d(cfg: Any, P: dict[str, Any], A: dict[str, Any], run: Any, ctx: dict[str, Any]) -> dict[str, Any]:
    import polars as pl

    from pathlib import Path

    rungs = P["rungs"]
    cols = next((rungs[r]["columns"] for r in ("B7", "B3", "B2") if rungs[r]["status"] == "eligible" and rungs[r].get("columns")), [])
    if not cols:
        raise ResearchBlocked(BLOCKED_MISSING_INPUT, "no Combine feature family eligible for the twin composite", missing=["B2 / B3 / B7"])
    comp = SC.composite(P["data"], cols)
    prim = A["primary"]
    outcome = prim["wide"].select("nfl_id", "y")
    pairs, long, summ = SC.twins_and_contrast(P["data"], comp, cfg.twin_covariates, cfg.twin_caliper, outcome, cfg.n_boot, cfg.seed)
    cases = SC.select_cases(long, cfg.n_case_studies)
    best = A.get("best") or "M1"
    written = []
    for k, c in enumerate(cases.iter_rows(named=True), start=1):
        md = SC.case_markdown(k, c, P["data"], comp, prim["wide"], cfg.twin_covariates, cols, best, cfg.pseudonym_salt, P["targets"][cfg.primary_target])
        written.append(run.write_text(md, f"reports/athlete_profiles/case_{k:02d}.md"))
    for name, df in (("twin_pairs", pairs), ("twin_contrast", summ)):
        ctx["outputs"] += run.table(df, name, "scouting")
    ctx["outputs"] += [str(p) for p in written]
    return {"pairs": pairs, "long": long, "summary": summ, "cases": cases, "files": [str(Path(p)) for p in written], "composite_columns": cols}


def run_e(cfg: Any, exp: Any, P: dict[str, Any], A: dict[str, Any], run: Any, ctx: dict[str, Any]) -> dict[str, Any]:
    import polars as pl

    col = P["targets"][cfg.primary_target]
    d = P["data"].filter(pl.col(col).cast(pl.Float64).is_finite())
    ctrl = RB.control_models(cfg, P["data"], cfg.primary_target, A["mods"], A["best"], exp.folds_by_athlete, A["blocks"], d, cfg.seed)
    controls = RB.run_controls(exp, P["data"], col, ctrl, cfg.n_boot, cfg.seed)
    rlabel = RB.random_label(exp, P["data"], col, A["mods"], A["best"], cfg.n_boot, cfg.seed)
    perm = RB.permutation(exp, P["data"], col, A["mods"], A["best"], cfg.n_perm, cfg.seed)
    expo = {"T1": "n_windows", "T6": "n_windows", "T3": "n_events_t3", "T4": "n_events_t4", "T5": "n_events_t5"}.get(cfg.primary_target)
    sens = RB.subgroup_sensitivity(A["primary"]["wide"], P["data"], A["best"], expo, max(200, cfg.n_boot // 5), cfg.seed) if A["best"] else pl.DataFrame()
    for name, df, fam in (("negative_controls", controls, "controls"), ("random_label", rlabel, "controls"), ("subgroup_sensitivity", sens, "robustness"),
                          ("permutation", pl.DataFrame([{k: str(v) for k, v in perm.items()}]), "robustness")):
        ctx["outputs"] += run.table(df, name, fam)
    return {"controls": controls, "random_label": rlabel, "permutation": perm, "sensitivity": sens}


def run_f(cfg: Any, P: dict[str, Any], A: dict[str, Any], B: dict[str, Any] | None, C: dict[str, Any] | None, D: dict[str, Any] | None, E: dict[str, Any] | None,
          run: Any, ctx: dict[str, Any]) -> dict[str, Any]:
    import polars as pl

    from pathlib import Path

    mde = None
    p = Path(cfg.m16_run_dir) / "tables" / "detectability" / "mde.parquet"
    if p.is_file():
        mde = pl.read_parquet(p)
    ev = EVD.evidence_table(P["rungs"], A["primary"], (E or {}).get("controls"), (E or {}).get("permutation") or {}, (B or {}).get("by_position"), mde, cfg)
    g = EVD.g18(ev)
    ctx["outputs"] += run.table(ev, "evidence", "synthesis") + run.table(g, "m18_gate", "synthesis")
    rep = RP.final_report(cfg, P["prereg"], P["rung_frame"], A["primary"], (B or {}).get("table"), (C or {}).get("long"), (C or {}).get("contrast"),
                          (D or {}).get("summary"), (E or {}).get("controls"), (E or {}).get("random_label"), (E or {}).get("permutation") or {},
                          (E or {}).get("sensitivity"), ev, g)
    exe = RP.executive_summary(g, ev, mde, cfg.primary_target)
    ctx["outputs"] += run.save_markdown(rep, "m18_final_report") + run.save_markdown(exe, "executive_summary")
    return {"evidence": ev, "gate": g, "report": rep, "executive": exe, "mde": mde}


__all__ = ["prepare", "run_a", "run_b", "run_c", "run_d", "run_e", "run_f", "target_columns"]
