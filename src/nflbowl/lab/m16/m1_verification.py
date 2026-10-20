"""T04 / E16.A0–A1 — official M1 contract audit and reproducibility verification.

Nothing here defines a new M1. The feature list is ``nflbowl.data.validation.M1_COLS`` (asserted equal to the research
engine's ``m12.advanced_evaluation.M1_COLS``); the model is the official ladder (``Experiment.transfer_predictions`` →
``ae.oof_ladder``: athlete outer folds, train-fold design, inner-CV ridge α, position dummies); the target is
``nfl.q90_shared`` only.

Three comparisons, all at ``cfg.reproduction_tol`` (default 1e-8), all on predictions (never on summary metrics alone):

1. **internal** — the ladder's ``pred_M1`` vs the baseline engine's independent refit ``pred_M1_check``
   (``transfer_fairness.baseline_predictions``; the M13 reproducibility rule);
2. **persisted** — the recomputed ``pred_M1`` and ``pred_M1+F0`` vs every compatible persisted official OOF record
   (``<experiment>/evaluation/records/transfer_oof__<hash>.parquet`` + ``.json`` sidecar) in the read-only M12 / M13 / M14
   experiment directories. Compatible = same bundle fingerprint, same science hash, same target, column present. An
   incompatible record is listed with its reason and never compared;
3. **contract** — fold map (ladder folds = experiment fold map = manifest folds), target alignment (ladder ``y`` = the
   athletes-table target), fold sizes, M1 coverage and units.

Every athlete-level difference is kept (``tables/baselines/m1_athlete_diffs``): maximum |Δ prediction|, mismatched
athletes and folds, athletes present in one source only. A mismatch is a visible FAILED_VALIDATION (after all evidence is
written), never a relaxed tolerance.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from ...artifacts.store import atomic_json
from ...data.validation import M1_COLS, TRANSFER_TARGET
from ...utils.reproducibility import utc_now
from ..status import BLOCKED_MISSING_INPUT, FAIL, NOT_EVALUATED, PASS, ResearchBlocked, ValidationFailed, gate_row
from .targets import TARGETS, frame_fingerprint, target_frame, target_summary

OPTIONAL_TESTS = ("three_cone", "short_shuttle", "vertical", "broad_jump", "bench", "arm_length", "hand_size", "wingspan")
RECORD_GLOB = "transfer_oof__*.parquet"
COMPARED = (("pred_M1", "pred_M1"), ("pred_M1+F0", "pred_M1+F0"))
REPRODUCED, REPRODUCED_INTERNAL_ONLY, MISMATCH = "REPRODUCED", "REPRODUCED_INTERNAL_ONLY", "MISMATCH"
NAN = float("nan")


@dataclass
class M1Verification:
    verdict: str
    summary: dict[str, Any]
    tables: dict[str, Any] = field(default_factory=dict)
    gate: Any = None

    @property
    def passed(self) -> bool:
        return self.verdict in (REPRODUCED, REPRODUCED_INTERNAL_ONLY)


# ====================================================================== contract (E16.A0)
def engine_m1_cols() -> tuple[str, ...] | None:
    """``M1_COLS`` of the research engine (``nflbdb …m12.advanced_evaluation``), or None when the backend is unavailable."""
    try:
        from ... import _backend

        return tuple(_backend.ae().M1_COLS)
    except Exception:                                               # noqa: BLE001 — reported as "not checked", never assumed equal
        return None


def m1_contract(athletes: Any, cfg: Any, bundle_dir: str = "") -> tuple[dict[str, Any], Any]:
    """(contract record, per-column table). Missing table / columns → BLOCKED_MISSING_INPUT naming the path and columns;
    a feature-list disagreement between the bundle contract and the engine → FAILED_VALIDATION."""
    import polars as pl

    where = str(Path(bundle_dir) / "athletes.parquet") if bundle_dir else "athletes.parquet"
    if athletes is None:
        raise ResearchBlocked(BLOCKED_MISSING_INPUT, f"{where} is absent: the official M1 features live there (re-export the bundle with include_transfer=True)",
                              missing=[where])
    eng = engine_m1_cols()
    if eng is not None and tuple(eng) != tuple(M1_COLS):
        raise ValidationFailed(f"M1 definition differs: bundle contract {M1_COLS} vs research engine {eng}", {"bundle": list(M1_COLS), "engine": list(eng)})
    missing = [c for c in ("nfl_id", "position_group", "linked", *M1_COLS) if c not in athletes.columns]
    if missing:
        raise ResearchBlocked(BLOCKED_MISSING_INPUT, f"{where} lacks the M1 contract column(s) {missing}", missing=[f"{where}:{c}" for c in missing])
    has_t = TRANSFER_TARGET in athletes.columns
    linked = athletes.filter(pl.col("linked").fill_null(False))
    eval_set = linked.filter(pl.col(TRANSFER_TARGET).is_not_null()) if has_t else linked
    rows = []
    for c in M1_COLS:
        v = athletes[c].cast(pl.Float64, strict=False)
        lo, hi = cfg.plausible_ranges.get(c, (-np.inf, np.inf))
        obs = v.drop_nulls().drop_nans()
        rows.append({"column": c, "dtype": str(athletes[c].dtype), "coverage_all": float(1 - v.is_null().mean()),
                     "coverage_linked": float(1 - linked[c].is_null().mean()) if linked.height else NAN,
                     "coverage_linked_with_target": float(1 - eval_set[c].is_null().mean()) if eval_set.height else NAN,
                     "min": float(obs.min()) if obs.len() else NAN, "median": float(obs.median()) if obs.len() else NAN, "max": float(obs.max()) if obs.len() else NAN,
                     "assumed_range": f"[{lo}, {hi}]", "implausible_share": float(((obs < lo) | (obs > hi)).mean()) if obs.len() else NAN})
    tab = pl.DataFrame(rows)
    optional = [c for c in OPTIONAL_TESTS if c in athletes.columns]
    rec = {"m1_cols": list(M1_COLS), "engine_m1_cols": list(eng) if eng else "not checked (backend unavailable)", "source": where,
           "n_athletes": athletes.height, "n_linked": linked.height, "target": TRANSFER_TARGET, "target_present": has_t,
           "n_linked_with_target": eval_set.height if has_t else 0, "optional_tests_present": optional,
           "optional_tests_absent": [c for c in OPTIONAL_TESTS if c not in athletes.columns],
           "units_note": "height in, weight lb, forty / ten s are ASSUMED (no data dictionary in the repository); implausible_share tests the assumption"}
    return rec, tab


# ====================================================================== persisted records
def discover_persisted_oof(dirs: Sequence[Path], bundle_fp: str, science_hash: str, target: str = TRANSFER_TARGET) -> Any:
    """Every ``evaluation/records/transfer_oof__*.parquet`` with its sidecar and a compatibility verdict."""
    import polars as pl

    rows = []
    for d in dirs:
        for p in sorted((Path(d) / "evaluation" / "records").glob(RECORD_GLOB)):
            side = p.with_suffix(".json")
            meta: dict[str, Any] = {}
            reasons = []
            try:
                meta = json.loads(side.read_text(encoding="utf-8")) if side.is_file() else {}
            except (OSError, json.JSONDecodeError) as e:
                reasons.append(f"sidecar unreadable: {e}")
            if not side.is_file():
                reasons.append("no .json sidecar: dependencies unknown")
            deps = meta.get("deps", {}) or {}
            try:
                cols = pl.read_parquet_schema(p)
                colnames = list(cols)
            except Exception as e:                                  # noqa: BLE001 — recorded
                colnames, reasons = [], [*reasons, f"parquet unreadable: {type(e).__name__}: {e}"]
            if deps.get("bundle") not in (None, bundle_fp):
                reasons.append(f"bundle {deps.get('bundle')} ≠ current {bundle_fp}")
            if deps.get("bundle") is None and side.is_file():
                reasons.append("sidecar records no bundle fingerprint")
            if deps.get("science") not in (None, science_hash):
                reasons.append(f"science {deps.get('science')} ≠ current {science_hash}")
            if meta.get("target") not in (None, target):
                reasons.append(f"target {meta.get('target')} ≠ {target}")
            if "pred_M1" not in colnames:
                reasons.append("no pred_M1 column")
            rows.append({"experiment": Path(d).name, "path": str(p), "deps_hash": p.stem.split("__")[-1], "bundle": deps.get("bundle"),
                         "science": deps.get("science"), "target": meta.get("target"), "models": json.dumps(meta.get("models", [])),
                         "written_at": meta.get("written_at"), "has_pred_M1": "pred_M1" in colnames, "has_pred_M1+F0": "pred_M1+F0" in colnames,
                         "compatible": not reasons, "reason": "; ".join(reasons)})
    return pl.DataFrame(rows, infer_schema_length=None) if rows else pl.DataFrame(
        schema={"experiment": pl.String, "path": pl.String, "deps_hash": pl.String, "bundle": pl.String, "science": pl.String, "target": pl.String,
                "models": pl.String, "written_at": pl.String, "has_pred_M1": pl.Boolean, "has_pred_M1+F0": pl.Boolean, "compatible": pl.Boolean,
                "reason": pl.String})


# ====================================================================== comparisons
def compare_predictions(ref: Any, cand: Any, pairs: Sequence[tuple[str, str]], tol: float, label: str) -> tuple[list[dict[str, Any]], Any]:
    """Athlete-level comparison of prediction columns of two OOF frames (nfl_id, fold, y, pred_…).

    Returns (one summary row per column pair, athlete-diff frame). A pair passes when: no athlete is in one source only,
    folds are identical, |Δy| ≤ tol and |Δpred| ≤ tol for every athlete."""
    import polars as pl

    out_rows, diffs = [], []
    for rc, cc in pairs:
        if rc not in ref.columns or cc not in cand.columns:
            out_rows.append({"comparison": label, "column": rc, "status": NOT_EVALUATED, "reason": f"column absent ({rc} in reference: {rc in ref.columns}; "
                             f"{cc} in candidate: {cc in cand.columns})"})
            continue
        a = ref.select(pl.col("nfl_id").cast(pl.Int64), pl.col("fold").cast(pl.Int64).alias("fold_ref"), pl.col("y").cast(pl.Float64).alias("y_ref"),
                       pl.col(rc).cast(pl.Float64).alias("pred_ref"))
        b = cand.select(pl.col("nfl_id").cast(pl.Int64), pl.col("fold").cast(pl.Int64).alias("fold_cand"), pl.col("y").cast(pl.Float64).alias("y_cand"),
                        pl.col(cc).cast(pl.Float64).alias("pred_cand"))
        j = a.join(b, on="nfl_id", how="full", coalesce=True)
        only_ref = j.filter(pl.col("pred_cand").is_null() & pl.col("pred_ref").is_not_null())
        only_cand = j.filter(pl.col("pred_ref").is_null() & pl.col("pred_cand").is_not_null())
        both = j.filter(pl.col("pred_ref").is_not_null() & pl.col("pred_cand").is_not_null()).with_columns(
            (pl.col("pred_cand") - pl.col("pred_ref")).abs().alias("abs_dpred"), (pl.col("y_cand") - pl.col("y_ref")).abs().alias("abs_dy"),
            (pl.col("fold_cand") != pl.col("fold_ref")).alias("fold_mismatch"))
        both = both.with_columns(((pl.col("abs_dpred") > tol) | (pl.col("abs_dy") > tol) | pl.col("fold_mismatch")).alias("mismatch"))
        dmax = float(both["abs_dpred"].max()) if both.height else NAN
        ymax = float(both["abs_dy"].max()) if both.height else NAN
        nfold = int(both["fold_mismatch"].sum()) if both.height else 0
        mism = both.filter(pl.col("mismatch"))
        per_fold = (both.group_by("fold_ref").agg(pl.col("abs_dpred").max().alias("max_abs_dpred"), pl.col("mismatch").sum().alias("n_mismatch"),
                                                  pl.len().alias("n")).sort("fold_ref").to_dicts()) if both.height else []
        ok = both.height > 0 and only_ref.height == 0 and only_cand.height == 0 and nfold == 0 and np.isfinite(dmax) and dmax <= tol and ymax <= tol
        out_rows.append({"comparison": label, "column": rc, "status": PASS if ok else FAIL, "n_compared": both.height, "athletes_only_reference": only_ref.height,
                         "athletes_only_candidate": only_cand.height, "max_abs_pred_diff": dmax, "max_abs_y_diff": ymax, "fold_mismatches": nfold,
                         "n_mismatched_athletes": mism.height, "mismatched_athletes": json.dumps(mism["nfl_id"].head(50).to_list()),
                         "mismatched_folds": json.dumps(sorted(set(mism["fold_ref"].drop_nulls().to_list()))), "per_fold": json.dumps(per_fold),
                         "tolerance": tol, "reason": "" if ok else "see athlete diffs"})
        diffs.append(pl.concat([both.select("nfl_id", "fold_ref", "fold_cand", "y_ref", "y_cand", "pred_ref", "pred_cand", "abs_dpred", "abs_dy", "mismatch"),
                                only_ref.select("nfl_id", "fold_ref", "fold_cand", "y_ref", "y_cand", "pred_ref", "pred_cand").with_columns(
                                    pl.lit(None, dtype=pl.Float64).alias("abs_dpred"), pl.lit(None, dtype=pl.Float64).alias("abs_dy"), pl.lit(True).alias("mismatch")),
                                only_cand.select("nfl_id", "fold_ref", "fold_cand", "y_ref", "y_cand", "pred_ref", "pred_cand").with_columns(
                                    pl.lit(None, dtype=pl.Float64).alias("abs_dpred"), pl.lit(None, dtype=pl.Float64).alias("abs_dy"), pl.lit(True).alias("mismatch"))],
                               how="vertical_relaxed").with_columns(pl.lit(label).alias("comparison"), pl.lit(rc).alias("column")))
    diff = pl.concat(diffs, how="vertical_relaxed") if diffs else None
    return out_rows, diff


def metric_rows(frame: Any, pred_cols: Sequence[str], source: str) -> list[dict[str, Any]]:
    """RMSE / MAE / R² / n overall and per fold for each prediction column (descriptive; equality is judged on predictions)."""
    rows = []
    y = frame["y"].to_numpy().astype(float)
    folds = frame["fold"].to_numpy()
    for c in pred_cols:
        if c not in frame.columns:
            continue
        p = frame[c].to_numpy().astype(float)
        ok = np.isfinite(p) & np.isfinite(y)
        for f in [None, *sorted(set(folds[ok].tolist()))]:
            m = ok if f is None else ok & (folds == f)
            if not m.any():
                continue
            e = p[m] - y[m]
            ss = float(((y[m] - y[m].mean()) ** 2).sum())
            rows.append({"source": source, "model": c.removeprefix("pred_"), "fold": "all" if f is None else str(int(f)), "n": int(m.sum()),
                         "rmse": float(np.sqrt((e ** 2).mean())), "mae": float(np.abs(e).mean()), "r2": float(1 - (e ** 2).sum() / ss) if ss > 0 else NAN})
    return rows


def fold_checks(ladder: Any, exp_folds: dict[int, int], manifest: Any, outer_folds: int, min_per_fold: int) -> list[dict[str, Any]]:
    """Ladder folds vs the experiment fold map vs the 12.0 manifest; every fold populated with ≥ ``min_per_fold`` athletes."""
    import polars as pl

    ids = [int(i) for i in ladder["nfl_id"].to_list()]
    lf = [int(f) for f in ladder["fold"].to_list()]
    vs_exp = sum(1 for i, f in zip(ids, lf, strict=True) if exp_folds.get(i) != f)
    man = dict(zip(*manifest.filter(pl.col("fold").is_not_null()).select("nfl_id", "fold").to_dict(as_series=False).values(), strict=True))
    vs_man = sum(1 for i, f in zip(ids, lf, strict=True) if i in man and int(man[i]) != f)
    not_in_man = sum(1 for i in ids if i not in man)
    sizes = {k: lf.count(k) for k in range(outer_folds)}
    small = {k: n for k, n in sizes.items() if n < min_per_fold}
    return [{"check": "ladder folds = Experiment.folds_by_athlete", "status": PASS if vs_exp == 0 else FAIL, "value": vs_exp, "detail": "athletes differing"},
            {"check": "ladder folds = 12.0 manifest folds", "status": PASS if vs_man == 0 else FAIL, "value": vs_man,
             "detail": f"athletes differing; {not_in_man} athlete(s) not in the manifest (fallback fold)"},
            {"check": f"every fold has ≥ {min_per_fold} evaluated athletes", "status": PASS if not small else FAIL, "value": json.dumps(sizes),
             "detail": f"small folds: {small}" if small else ""},
            {"check": "one fold per athlete", "status": PASS if len(set(ids)) == len(ids) else FAIL, "value": len(ids) - len(set(ids)), "detail": "duplicated athletes"}]


def target_alignment(ladder: Any, athletes: Any, tol: float) -> dict[str, Any]:
    """The ladder's ``y`` must equal the athletes-table official target for every athlete."""
    import polars as pl

    t = athletes.select(pl.col("nfl_id").cast(pl.Int64), pl.col(TRANSFER_TARGET).cast(pl.Float64).alias("t"))
    j = ladder.select(pl.col("nfl_id").cast(pl.Int64), pl.col("y").cast(pl.Float64)).join(t, on="nfl_id", how="left")
    missing = int(j["t"].is_null().sum())
    d = j.drop_nulls()
    dmax = float((d["y"] - d["t"]).abs().max()) if d.height else NAN
    ok = missing == 0 and np.isfinite(dmax) and dmax <= tol
    return {"check": f"ladder y = athletes.{TRANSFER_TARGET}", "status": PASS if ok else FAIL, "value": dmax,
            "detail": f"{missing} ladder athlete(s) without the target in athletes.parquet"}


# ====================================================================== orchestration (T04)
def _nanmax(values: Any) -> float:
    """Max over finite readings; NaN when there is none (an all-NaN comparison is reported, never read as 0)."""
    v = [float(x) for x in values if x is not None and np.isfinite(float(x))]
    return max(v) if v else NAN


def verification_signature(cfg: Any, exp: Any) -> dict[str, Any]:
    return {"bundle": exp.bundle.fingerprint, "science": exp.cfg.science_hash(), "config": cfg.fingerprint(),
            "persisted": sorted(str(p) for d in cfg.experiment_dirs() for p in (Path(d) / "evaluation" / "records").glob(RECORD_GLOB))}


TABLE_NAMES = ("m1_contract_columns", "m1_reproducibility", "m1_checks", "m1_metrics", "m1_persisted_records")


def verify_m1(cfg: Any, run: Any, exp: Any, ctx: dict[str, Any] | None = None) -> M1Verification:
    """Run the complete T04 verification, persist every table and the audit report, then raise ``ValidationFailed`` if the
    official M1 is not reproduced (the tables stay on Drive as evidence). Reuses saved tables when their input signature
    (bundle, science, config, persisted-record list) is unchanged (→ ``ctx['cache'] = 'hit'``)."""
    import polars as pl

    from .. import transfer_fairness as TF

    ctx = ctx if ctx is not None else {}
    sig = verification_signature(cfg, exp)
    hit = run.cached("baselines", list(TABLE_NAMES), signature=sig, key="baselines/m1_verification") if cfg.reuse_artifacts else None
    summary_path = run.root / "manifests" / "m1_verification.json"
    if hit is not None and summary_path.is_file():
        ctx["cache"] = "hit"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        v = M1Verification(summary["verdict"], summary, hit, gate=gate_g16b(summary))
        if not v.passed:
            raise ValidationFailed(f"official M1 NOT reproduced (cached result): {summary.get('failures')}", summary)
        return v

    bundle = exp.bundle
    contract, contract_cols = m1_contract(bundle.athletes, cfg, str(bundle.root))
    run.table(contract_cols, "m1_contract_columns", "baselines")
    run.table(pl.DataFrame([{"key": k, "value": json.dumps(v, default=str)} for k, v in contract.items()]), "m1_contract", "cohort")
    atomic_json(contract, run.root / "manifests" / "m1_contract.json", overwrite=True)

    ladder = exp.transfer_predictions(models=[], reuse=cfg.reuse_artifacts)           # official ladder: M1 · M1+F0 (· M1+F1_TCN)
    baseline, _frame = TF.baseline_predictions(exp)                                      # POS · MEAN · independent M1 refit
    run.data_file(ladder, "predictions/m1_ladder_oof", {"bundle": bundle.fingerprint, "science": exp.cfg.science_hash()})
    run.data_file(baseline, "predictions/m1_baseline_oof", {"bundle": bundle.fingerprint, "science": exp.cfg.science_hash()})

    tol = cfg.reproduction_tol
    rep_rows, diffs = compare_predictions(ladder, baseline, [("pred_M1", "pred_M1_check")], tol, "internal: ladder M1 vs baseline-engine M1 refit")
    old = TF.reproducibility_check(ladder, baseline, tol)                                # the M13 rule, kept as an independent second reading
    rep_rows.append({"comparison": "internal (transfer_fairness.reproducibility_check)", "column": "pred_M1",
                     "status": PASS if old["status"] == "pass" else FAIL, "max_abs_pred_diff": old["max_abs_pred_diff"], "max_abs_y_diff": old["max_abs_y_diff"],
                     "athletes_only_reference": old["athletes_in_one_only"], "tolerance": tol})
    diff_parts = [diffs] if diffs is not None else []

    records = discover_persisted_oof(cfg.experiment_dirs(), bundle.fingerprint, exp.cfg.science_hash())
    run.table(records, "m1_persisted_records", "baselines")
    compatible = records.filter(pl.col("compatible")) if records.height else records
    metric_list = metric_rows(ladder, ["pred_M1", "pred_M1+F0"], "recomputed (M16)") + metric_rows(baseline, ["pred_POS", "pred_MEAN", "pred_M1_check"], "baseline engine (M16)")
    for r in compatible.iter_rows(named=True):
        persisted = pl.read_parquet(r["path"])
        rows, d = compare_predictions(persisted, ladder, COMPARED, tol, f"persisted {r['experiment']}/{Path(r['path']).name} vs recomputed")
        rep_rows += rows
        if d is not None:
            diff_parts.append(d)
        metric_list += metric_rows(persisted, ["pred_M1", "pred_M1+F0"], f"persisted {r['experiment']}")
    if not compatible.height:
        rep_rows.append({"comparison": "persisted official OOF vs recomputed", "column": "pred_M1", "status": NOT_EVALUATED,
                         "reason": f"{BLOCKED_MISSING_INPUT}: no compatible transfer_oof record in {[str(d) for d in cfg.experiment_dirs()] or 'any configured experiment directory'}"
                                   + (f" ({records.height} incompatible record(s) listed in m1_persisted_records)" if records.height else "")})
    checks = [*fold_checks(ladder, exp.folds_by_athlete, bundle.folds, exp.cfg.science.outer_folds, cfg.min_athletes_per_fold),
              target_alignment(ladder, bundle.athletes, tol)]
    rep = pl.DataFrame(rep_rows, infer_schema_length=None)
    chk = pl.DataFrame(checks, infer_schema_length=None).with_columns(pl.col("value").cast(pl.String))
    met = pl.DataFrame(metric_list, infer_schema_length=None)
    run.table(rep, "m1_reproducibility", "baselines")
    run.table(chk, "m1_checks", "baselines")
    run.table(met, "m1_metrics", "baselines")
    if diff_parts:
        run.table(pl.concat(diff_parts, how="vertical_relaxed"), "m1_athlete_diffs", "baselines")
    official = target_frame(TARGETS["T0"], bundle.athletes)
    t0 = target_summary(TARGETS["T0"], official)

    internal_ok = all(r["status"] == PASS for r in rep_rows if str(r["comparison"]).startswith("internal"))
    persisted = [r for r in rep_rows if str(r["comparison"]).startswith("persisted ") and r["status"] != NOT_EVALUATED]
    persisted_ok = bool(persisted) and all(r["status"] == PASS for r in persisted)
    checks_ok = all(c["status"] == PASS for c in checks)
    failures = [f"{r['comparison']} [{r['column']}]" for r in rep_rows if r["status"] == FAIL] + [c["check"] for c in checks if c["status"] == FAIL]
    if not internal_ok or not checks_ok or (persisted and not persisted_ok and cfg.require_persisted_match):
        verdict = MISMATCH
    elif persisted_ok:
        verdict = REPRODUCED
    else:
        verdict = REPRODUCED_INTERNAL_ONLY
    summary = {"verdict": verdict, "tolerance": tol, "bundle": bundle.fingerprint, "science": exp.cfg.science_hash(), "target": TRANSFER_TARGET,
               "target_summary": t0, "n_ladder_athletes": ladder.height, "internal_ok": internal_ok, "checks_ok": checks_ok,
               "n_persisted_records": records.height, "n_compatible_records": compatible.height, "persisted_ok": persisted_ok if persisted else None,
               "max_abs_pred_diff_internal": _nanmax(r.get("max_abs_pred_diff") for r in rep_rows if str(r["comparison"]).startswith("internal")),
               "max_abs_pred_diff_persisted": _nanmax(r.get("max_abs_pred_diff") for r in persisted) if persisted else None,
               "failures": failures, "ladder_fingerprint": frame_fingerprint(ladder.select("nfl_id", "fold", "y", "pred_M1")), "written_at": utc_now()}
    atomic_json(summary, summary_path, overwrite=True)
    run.set_signature("baselines/m1_verification", sig)
    gate = gate_g16b(summary)
    run.table(gate, "g16b_gate", "synthesis")
    run.save_markdown(report_markdown(summary, contract, rep, chk, records), "m1_audit_report")
    tables = {"m1_contract_columns": contract_cols, "m1_reproducibility": rep, "m1_checks": chk, "m1_metrics": met, "m1_persisted_records": records}
    v = M1Verification(verdict, summary, tables, gate)
    if not v.passed:
        raise ValidationFailed(f"official M1 NOT reproduced at tolerance {tol}: {failures}", summary)
    return v


def gate_g16b(summary: dict[str, Any]) -> Any:
    """G16-B rows. PASS only from computed comparisons; no persisted record → that criterion stays NOT_EVALUATED."""
    import polars as pl

    tol = summary.get("tolerance")
    rows = [gate_row("M1 internal reproduction (ladder vs refit)", summary.get("max_abs_pred_diff_internal"), tol, bool(summary.get("internal_ok"))),
            gate_row("fold / target contract", 1.0 if summary.get("checks_ok") else 0.0, 1.0, bool(summary.get("checks_ok"))),
            gate_row("M1 and M1+F0 vs persisted official OOF", summary.get("max_abs_pred_diff_persisted"), tol, summary.get("persisted_ok"),
                     note="" if summary.get("persisted_ok") is not None else "no compatible persisted record: send the M12/M13 experiment records")]
    statuses = [r["status"] for r in rows]
    overall = FAIL if FAIL in statuses else (PASS if all(s == PASS for s in statuses) else NOT_EVALUATED)
    rows.append({"criterion": "G16-B overall", "value": None, "ci_lo": None, "ci_hi": None, "threshold": None, "status": overall,
                 "note": f"verdict {summary.get('verdict')}"})
    return pl.DataFrame(rows, infer_schema_length=None)


def report_markdown(summary: dict[str, Any], contract: dict[str, Any], rep: Any, chk: Any, records: Any) -> str:
    """The M1 audit report — every number read from the computed tables / summary."""
    def tbl(df: Any, cols: Sequence[str]) -> list[str]:
        cols = [c for c in cols if c in df.columns]
        out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
        for r in df.select(cols).iter_rows():
            out.append("| " + " | ".join("" if v is None else (f"{v:.3g}" if isinstance(v, float) else str(v)) for v in r) + " |")
        return out

    lines = ["# Official M1 audit (T04 / E16.A0–A1)", "", f"Written {summary['written_at']} · bundle `{summary['bundle']}` · science `{summary['science']}` · "
             f"target `{summary['target']}` · tolerance {summary['tolerance']}", "", f"**Verdict: {summary['verdict']}**", "",
             "## M1 contract", "", f"- Features: {', '.join(contract['m1_cols'])} (engine: {contract['engine_m1_cols']})",
             f"- Athletes {contract['n_athletes']}, linked {contract['n_linked']}, linked with target {contract['n_linked_with_target']}",
             f"- Optional official tests present: {contract['optional_tests_present'] or 'none'}", f"- {contract['units_note']}", "",
             "## Prediction comparisons", "", *tbl(rep, ["comparison", "column", "status", "n_compared", "max_abs_pred_diff", "max_abs_y_diff", "fold_mismatches",
                                                       "n_mismatched_athletes", "athletes_only_reference", "athletes_only_candidate", "reason"]), "",
             "## Contract checks", "", *tbl(chk, ["check", "status", "value", "detail"]), "",
             "## Persisted official OOF records found", ""]
    lines += tbl(records, ["experiment", "deps_hash", "compatible", "reason"]) if records.height else ["None found in the configured experiment directories."]
    lines += ["", "## Reading", "", "- REPRODUCED: the recomputed official M1 equals the persisted official predictions and its own independent refit.",
              "- REPRODUCED_INTERNAL_ONLY: equal to its own refit; no compatible persisted record was available to compare (G16-B stays NOT_EVALUATED).",
              "- MISMATCH: see `tables/baselines/m1_athlete_diffs` (athletes, folds, |Δ|); M18 rungs B1+ stay blocked until resolved.",
              "", "The exploratory target `nfl_speed_q90_median` is audited separately (`exploratory_target_audit`) and never compared with this one."]
    return "\n".join(lines)


def exploratory_target_audit(cfg: Any, athletes: Any, run: Any | None = None) -> dict[str, Any]:
    """Coverage / fingerprint of the exploratory target T0x, side by side with T0 — descriptive only, never compared on RMSE.
    Absent → BLOCKED_MISSING_INPUT naming where it was looked for."""
    import polars as pl

    t0 = target_summary(TARGETS["T0"], target_frame(TARGETS["T0"], athletes))
    tx_frame = target_frame(TARGETS["T0x"], athletes, cfg.exploratory_target_path)
    tx = target_summary(TARGETS["T0x"], tx_frame)
    official_ids = set(int(i) for i in target_frame(TARGETS["T0"], athletes)["nfl_id"].to_list())
    tx_ids = set(int(i) for i in tx_frame["nfl_id"].to_list())
    out = {"T0": t0, "T0x": tx, "overlap_athletes": len(official_ids & tx_ids), "only_T0": len(official_ids - tx_ids), "only_T0x": len(tx_ids - official_ids),
           "note": "different targets: units may agree but definitions, cohorts and scales differ; RMSEs are never compared"}
    if run is not None:
        run.table(pl.DataFrame([{"target_id": k, **{kk: json.dumps(vv) if isinstance(vv, (list, dict)) else vv for kk, vv in v.items()}}
                                for k, v in (("T0", t0), ("T0x", tx))], infer_schema_length=None), "target_registry_t0_t0x", "target")
        atomic_json(out, run.root / "manifests" / "exploratory_target_audit.json", overwrite=True)
    return out


__all__ = ["COMPARED", "M1Verification", "MISMATCH", "OPTIONAL_TESTS", "REPRODUCED", "REPRODUCED_INTERNAL_ONLY", "compare_predictions",
           "discover_persisted_oof", "engine_m1_cols", "exploratory_target_audit", "fold_checks", "gate_g16b", "m1_contract", "metric_rows",
           "report_markdown", "target_alignment", "verify_m1"]
