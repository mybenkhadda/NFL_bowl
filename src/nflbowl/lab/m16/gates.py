"""G16 — verified baselines and cross-domain validity, from computed M16 tables only.

    G16-A  target reliability       PASS when ≥ 1 task target passes ``target_reliability.gate_a`` (T0 stays primary unless the
                                    analyst records otherwise); FAIL when every computed task target fails; NOT_EVALUATED otherwise
    G16-B  M1 reproducibility       the overall row of ``g16b_gate`` (M1 verification)
    G16-C  incremental information  on the primary target: PASS if M1+simple or M1+M14_reliable is ``validated`` vs M1;
                                    PRECISE_NULL if both comparisons satisfy the precise-null rule; INCONCLUSIVE otherwise
    G16-D  cross-domain similarity  PASS when some eligible representation has (same-position MRR vs F0 CI > 0, Holm p ≤ 0.05)
                                    AND (all-pool top-1 above position-only, CI > 0) AND a CEM-matched same-position MRR not below
                                    the original by more than half the F0 margin; FAIL when tests exist and none passes
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..status import FAIL, NOT_EVALUATED, PASS


def g16(gate_a: Any, g16b: Any, status_primary: Any, paired_primary: Any, primary_tests: Any, cem: Any) -> Any:
    import polars as pl

    rows = []
    # A
    if gate_a is None or not gate_a.height:
        rows.append({"criterion": "G16-A target reliability", "status": NOT_EVALUATED, "note": "no reliability table"})
    else:
        task = gate_a.filter(pl.col("target_id") != "T0")
        st = task["status"].to_list()
        s = PASS if PASS in st else (FAIL if st and all(x == FAIL for x in st) else NOT_EVALUATED)
        rows.append({"criterion": "G16-A target reliability", "status": s, "note": f"passing task targets: {task.filter(pl.col('status') == PASS)['target_id'].to_list()}"})
    # B
    if g16b is None or not g16b.height:
        rows.append({"criterion": "G16-B M1 reproducibility", "status": NOT_EVALUATED, "note": "no M1 verification"})
    else:
        o = g16b.filter(pl.col("criterion") == "G16-B overall")
        rows.append({"criterion": "G16-B M1 reproducibility", "status": o["status"][0] if o.height else NOT_EVALUATED, "note": o["note"][0] if o.height else ""})
    # C
    if status_primary is None or not status_primary.height:
        rows.append({"criterion": "G16-C incremental information", "status": NOT_EVALUATED, "note": "no baseline ladder on the primary target"})
    else:
        sel = status_primary.filter(pl.col("model").is_in(["M1+simple", "M1+M14_reliable"]) & (pl.col("reference") == "M1"))
        pn = paired_primary.filter(pl.col("model").is_in(["M1+simple", "M1+M14_reliable"]) & (pl.col("reference") == "M1")) if paired_primary is not None else None
        if (sel["status"] == "validated").any():
            s = PASS
        elif pn is not None and pn.height and "precise_null" in pn.columns and bool(pn["precise_null"].all()):
            s = "PRECISE_NULL"
        elif sel.height:
            s = "INCONCLUSIVE"
        else:
            s = NOT_EVALUATED
        rows.append({"criterion": "G16-C incremental information", "status": s, "note": "; ".join(f"{r['model']}: {r['status']}" for r in sel.iter_rows(named=True))})
    # D
    if primary_tests is None or not primary_tests.height:
        rows.append({"criterion": "G16-D cross-domain similarity", "status": NOT_EVALUATED, "note": "no retrieval tests (no eligible representation?)"})
    else:
        ok_reps = []
        for rep in primary_tests["rep"].unique().to_list():
            t = primary_tests.filter(pl.col("rep") == rep)
            m = t.filter(pl.col("test").str.starts_with("same_position"))
            pchk = t.filter(pl.col("test").str.starts_with("all-pool"))
            a_ok = bool(m.height and m["lo"][0] > 0 and (m["p_holm"][0] is not None and m["p_holm"][0] <= 0.05))
            b_ok = bool(pchk.height and pchk["lo"][0] > 0)
            c_ok = True
            if cem is not None and cem.height:
                c = cem.filter(pl.col("rep") == rep).group_by("condition").agg(pl.col("same_position_mrr").mean())
                o = c.filter(pl.col("condition") == "original")["same_position_mrr"]
                mt = c.filter(pl.col("condition") == "cem_matched")["same_position_mrr"]
                if o.len() and mt.len() and m.height:
                    c_ok = bool(np.isfinite(mt[0]) and (o[0] - mt[0]) <= 0.5 * float(m["estimate"][0]))
            if a_ok and b_ok and c_ok:
                ok_reps.append(rep)
        rows.append({"criterion": "G16-D cross-domain similarity", "status": PASS if ok_reps else FAIL, "note": f"passing: {ok_reps}"})
    return pl.DataFrame(rows, infer_schema_length=None)


__all__ = ["g16"]
