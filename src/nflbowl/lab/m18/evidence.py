"""E18.F1 / G18 — evidence registry per feature family and the scouting-relevance verdict (gates document, G18 1–6).

    1 measurement gate     B3: M14 ``validated`` (eligibility rule) · B4/B5: G15 GO · B6: both · B7: G17 validated ·
                           B2: NOT_EVALUATED (no measurement gate is defined for the simple statistics — stated, never assumed)
    2 predictive           ``transfer_status`` = validated vs M1 on the primary target (Holm over the pre-registered family)
    3 practical margin     ΔRMSE ≥ ``margin_pct`` % of RMSE(M1)
    4 negative controls    every control fitted for this family has ``control_status`` PASS
    5 permutation          refit permutation p ≤ 0.05 (computed for the best rung only; others NOT_EVALUATED)
    6 consistency          ≥ 4/5 folds better AND positive point estimate in ≥ 3 positions with n ≥ 30
Verdict: scouting_relevant (1–6) · promising_needs_replication (1–3 and one of 4–6 not PASS) · precise_null · inconclusive
(otherwise) · blocked (rung not eligible). Replication on new draft classes is ALWAYS listed as required.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..status import FAIL, NOT_EVALUATED, PASS
from .ladder import NAMES

MEASUREMENT = {"B2": (NOT_EVALUATED, "no measurement gate defined for simple statistics"), "B3": (PASS, "M14 validated metrics only"),
               "B4": (PASS, "G15 GO"), "B5": (PASS, "G15 GO"), "B6": (PASS, "G15 GO (both views)"), "B7": (PASS, "G17 validated signatures only")}


def evidence_table(rungs: dict[str, dict[str, Any]], A: dict[str, Any], controls: Any, perm: dict[str, Any], by_pos: Any, mde: Any, cfg: Any) -> Any:
    import polars as pl

    status, paired = A.get("status"), A.get("paired")
    stab = {}
    folds = A.get("folds")
    if folds is not None and folds.height and "delta_rmse_vs_M1" in folds.columns:
        for m, g in folds.group_by("model"):
            stab[m[0]] = int((g["delta_rmse_vs_M1"] > 0).sum())
    rows = []
    for r in ("B2", "B3", "B4", "B5", "B6", "B7"):
        name = NAMES[r]
        if rungs[r]["status"] != "eligible":
            rows.append({"rung": r, "model": name, "verdict": "blocked", "reason": rungs[r]["reason"]})
            continue
        c1 = MEASUREMENT[r]
        st = status.filter(pl.col("model") == name) if status is not None and status.height else None
        pr = paired.filter(pl.col("model") == name) if paired is not None and paired.height else None
        c2 = (PASS if st["status"][0] == "validated" else FAIL) if st is not None and st.height else NOT_EVALUATED
        c3 = (PASS if bool(pr["exceeds_practical_margin"][0]) else FAIL) if pr is not None and pr.height and "exceeds_practical_margin" in pr.columns else NOT_EVALUATED
        ctrl = controls.filter(pl.col("model").str.contains(name, literal=True) | (pl.col("model") == "CTRL_random_embedding")) if controls is not None and controls.height else None
        if r in ("B4", "B5", "B6") or (perm or {}).get("model") == name:
            c4 = (PASS if ctrl is not None and ctrl.height and (ctrl["control_status"] == "PASS").all() else (FAIL if ctrl is not None and ctrl.height else NOT_EVALUATED))
        else:
            c4 = NOT_EVALUATED
        c5 = (PASS if (perm.get("p_value") is not None and np.isfinite(perm["p_value"]) and perm["p_value"] <= 0.05) else FAIL) if (perm or {}).get("model") == name else NOT_EVALUATED
        pos_ok = None
        if by_pos is not None and by_pos.height and "model" in by_pos.columns:
            bp = by_pos.filter((pl.col("model") == name) & (pl.col("reference") == "M1") & (pl.col("n_athletes") >= cfg.min_position_n))
            pos_ok = int((bp["delta_rmse"] > 0).sum()) if bp.height else None
        fb = stab.get(name)
        c6 = (PASS if (fb is not None and fb >= 4 and pos_ok is not None and pos_ok >= 3) else FAIL) if (fb is not None and pos_ok is not None) else NOT_EVALUATED
        crit = {"1 measurement": c1[0], "2 predictive validated": c2, "3 practical margin": c3, "4 negative controls": c4, "5 permutation": c5, "6 consistency": c6}
        precise = bool(pr["precise_null"][0]) if pr is not None and pr.height and "precise_null" in pr.columns else False
        if all(v == PASS for v in crit.values()):
            verdict = "scouting_relevant"
        elif all(crit[k] == PASS for k in ("1 measurement", "2 predictive validated", "3 practical margin")):
            verdict = "promising_needs_replication"
        elif precise:
            verdict = "precise_null"
        else:
            verdict = "inconclusive"
        rows.append({"rung": r, "model": name, **{f"c{k}": v for k, v in crit.items()}, "measurement_note": c1[1],
                     "delta_rmse": float(pr["delta_rmse"][0]) if pr is not None and pr.height else None, "delta_lo": float(pr["delta_lo"][0]) if pr is not None and pr.height else None,
                     "delta_hi": float(pr["delta_hi"][0]) if pr is not None and pr.height else None, "delta_pct": float(pr["delta_pct"][0]) if pr is not None and pr.height else None,
                     "folds_better": fb, "positions_positive": pos_ok, "verdict": verdict, "replication_required": True, "reason": ""})
    out = pl.DataFrame(rows, infer_schema_length=None)
    if mde is not None and mde.height:
        out = out.with_columns(pl.lit(mde.to_dicts().__repr__()[:500]).alias("mde_context"))
    return out


def g18(evidence: Any) -> Any:
    import polars as pl

    if evidence is None or not evidence.height:
        return pl.DataFrame({"criterion": ["G18 overall"], "status": [NOT_EVALUATED], "note": ["no evidence table"]})
    v = evidence["verdict"].to_list()
    overall = "scouting_relevant" if "scouting_relevant" in v else ("promising_needs_replication" if "promising_needs_replication" in v else
                                                                   ("precise_null" if v and all(x in ("precise_null", "blocked") for x in v) and "precise_null" in v else
                                                                    ("blocked" if all(x == "blocked" for x in v) else "inconclusive")))
    rows = [{"criterion": f"G18 {r['rung']} {r['model']}", "status": r["verdict"], "note": r.get("reason") or ""} for r in evidence.iter_rows(named=True)]
    rows.append({"criterion": "G18 overall", "status": overall, "note": "replication on independent draft classes required in every case"})
    return pl.DataFrame(rows, infer_schema_length=None)


__all__ = ["MEASUREMENT", "evidence_table", "g18"]
