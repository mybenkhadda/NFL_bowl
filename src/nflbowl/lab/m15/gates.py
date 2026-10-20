"""G15 — representation readiness, computed only from the M15 tables (``docs/research/M15_M18_SCIENTIFIC_GATES.md``).

Per representation (arch, d):
    G15.a  physical preservation   paired (model − PCA) upper CI of the per-sequence mean speed AND a_tan RMSE ≤ +5 % of PCA's mean
    G15.b  event preservation      mean event-F1 difference vs PCA ≥ −0.05 AND median braking-onset error ≤ 0.2 s
    G15.c  repeatability           median per-dimension ICC ≥ 0.4 AND share of dimensions ≥ 0.6 is ≥ 25 % (forty, and an agility drill
                                   when it has enough athletes)
    G15.d  no collapse             effective rank / d ≥ 0.25 AND no collapsed dimension (median over folds)
    G15.e  seed stability          Procrustes cross-seed r ≥ 0.8 (needs ≥ 2 seeds; otherwise NOT_EVALUATED)
    G15.f  robustness (reported)   self-retrieval top-1 at 10 % dropout ≥ 0.9

Decision: GO (a–e PASS) · GO_DESCRIPTIVE (a, b, d PASS; c or e FAIL) · NO_GO (a, b or d FAIL) · NOT_EVALUATED (an input of
a–d missing). PCA is the reference of a and b, so its a / b rows read PASS with the note "reference".
STFT models have no time-domain reconstruction: their a / b are NOT_EVALUATED (spectrogram loss is reported instead),
so an STFT model can at most be GO_DESCRIPTIVE unless the phase variant adds a time-domain path (not in this release).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..status import FAIL, NOT_EVALUATED, PASS, gate_row

NAN = float("nan")


def _get(df: Any, **eq: Any) -> Any:
    import polars as pl

    if df is None or not df.height:
        return df
    e = pl.lit(True)
    for k, v in eq.items():
        if k in df.columns:
            e = e & (pl.col(k) == v)
    return df.filter(e)


def g15(paired: Any, pres_summary: Any, icc_sum: Any, geom: Any, seeds: Any, robust: Any, archs: list[str], dims: list[int]) -> Any:
    import polars as pl

    rows = []
    for arch in archs:
        for d in dims:
            lab = {"arch": arch, "d": d}
            # a / b
            if arch == "pca":
                ra = gate_row("G15.a preservation", 0.0, 0.05, True, note="reference")
                rb = gate_row("G15.b event preservation", 0.0, -0.05, True, note="reference")
            else:
                p = _get(paired, arch=arch, d=d)
                if p is None or not p.height:
                    ra = gate_row("G15.a preservation", None, 0.05, None, note="no paired preservation table (or no time-domain reconstruction)")
                    rb = gate_row("G15.b event preservation", None, -0.05, None)
                else:
                    rel_hi = []
                    for met in ("rmse_speed", "rmse_atan"):
                        q = p.filter(pl.col("metric") == met)
                        if q.height:
                            rel_hi.append(float((q["hi"] / q["ref_mean"]).max()))
                    va = max(rel_hi) if len(rel_hi) == 2 else None
                    ra = gate_row("G15.a preservation", va, 0.05, (va <= 0.05) if va is not None and np.isfinite(va) else None, note="max upper CI / PCA mean")
                    f1 = p.filter(pl.col("metric") == "event_f1")
                    on = _get(pres_summary, arch=arch, d=d)
                    onset = float(on["brake_onset_err_s"].median()) if on is not None and on.height and "brake_onset_err_s" in on.columns else None
                    vb = float(f1["mean_diff"].min()) if f1.height else None
                    okb = (vb >= -0.05 and onset is not None and np.isfinite(onset) and onset <= 0.2) if vb is not None and np.isfinite(vb) else None
                    rb = gate_row("G15.b event preservation", vb, -0.05, okb, note=f"median braking-onset error {onset}")
            # c
            ic = _get(icc_sum, arch=arch, d=d)
            if ic is None or not ic.height:
                rc = gate_row("G15.c repeatability", None, 0.4, None, note="no per-dimension ICC with enough athletes")
            else:
                fo = ic.filter(pl.col("kind") == "forty")
                ag = ic.filter(pl.col("kind") != "forty")
                med = float(fo["median_dim_icc"].median()) if fo.height else NAN
                share = float(fo["share_dims_ge_0_6"].median()) if fo.height else NAN
                ag_ok = (not ag.height) or (float(ag["median_dim_icc"].median()) >= 0.4 and float(ag["share_dims_ge_0_6"].median()) >= 0.25)
                okc = (med >= 0.4 and share >= 0.25 and ag_ok) if np.isfinite(med) else None
                rc = gate_row("G15.c repeatability", med, 0.4, okc, note=f"forty share ≥ 0.6: {share:.2f}; agility ok: {ag_ok}")
            # d
            g = _get(geom, arch=arch, d=d)
            if g is None or not g.height or "effective_rank_over_d" not in g.columns:
                rd = gate_row("G15.d no collapse", None, 0.25, None)
            else:
                er = float(g["effective_rank_over_d"].median())
                col = float(g["collapsed_dim_share"].max())
                rd = gate_row("G15.d no collapse", er, 0.25, bool(er >= 0.25 and col == 0.0), note=f"max collapsed share {col:.2f}")
            # e
            s = _get(seeds, arch=arch, d=d)
            if s is None or not s.height:
                re_ = gate_row("G15.e seed stability", None, 0.8, None, note="needs ≥ 2 training seeds")
            else:
                sr = float(s["seed_r"].median())
                re_ = gate_row("G15.e seed stability", sr, 0.8, bool(sr >= 0.8) if np.isfinite(sr) else None)
            # f
            rb_ = _get(robust, arch=arch, d=d)
            q = rb_.filter(pl.col("condition") == "dropout_0.1") if rb_ is not None and rb_.height else None
            vf = float(q["self_top1"].median()) if q is not None and q.height else None
            rf = gate_row("G15.f robustness (reported)", vf, 0.9, (vf >= 0.9) if vf is not None else None)
            crit = [ra, rb, rc, rd, re_, rf]
            st = {r["criterion"][:5]: r["status"] for r in crit}
            if any(st[k] == NOT_EVALUATED for k in ("G15.a", "G15.b", "G15.d")):
                decision = NOT_EVALUATED
            elif any(st[k] == FAIL for k in ("G15.a", "G15.b", "G15.d")):
                decision = "NO_GO"
            elif st["G15.c"] == PASS and st["G15.e"] == PASS:
                decision = "GO"
            elif st["G15.c"] == NOT_EVALUATED or st["G15.e"] == NOT_EVALUATED:
                decision = "GO_DESCRIPTIVE_PENDING"
            else:
                decision = "GO_DESCRIPTIVE"
            rows += [{**lab, **r} for r in crit] + [{**lab, "criterion": "G15 decision", "value": None, "ci_lo": None, "ci_hi": None, "threshold": None,
                                                    "status": decision, "note": ""}]
    return pl.DataFrame(rows, infer_schema_length=None)


def advancing(gate: Any, allow_descriptive: bool = False) -> list[tuple[str, int]]:
    """(arch, d) representations allowed into M16-B / M18 B4–B6 as predictors (GO only unless ``allow_descriptive``)."""
    import polars as pl

    ok = ("GO", "GO_DESCRIPTIVE") if allow_descriptive else ("GO",)
    t = gate.filter((pl.col("criterion") == "G15 decision") & pl.col("status").is_in(list(ok)))
    return [(r["arch"], int(r["d"])) for r in t.iter_rows(named=True)]


__all__ = ["advancing", "g15"]
