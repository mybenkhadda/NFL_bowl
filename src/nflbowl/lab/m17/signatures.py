"""E17.E1 + G17 — candidate athletic signatures and their validation status.

Candidate = the Combine context-adjusted athlete effect (BLUP, ``hierarchical.athlete_effects``) of one event metric,
computed for every athlete with parameters from the OTHER folds (column ``sig_<metric>``). Criteria (gates document):

    G17.a  repeatability after context   athlete share of residual variance (= ICC of one adjusted event) ≥ icc_hi with its
                                         minimum over folds ≥ icc_lo
    G17.b  signal > noise                athlete share of TOTAL variance (incl. context) ≥ athlete_share_min, minimum over folds ≥ 0.25
    G17.c  novelty                       OOF R² of the signature from M1 + position ≤ novelty_max_r2 (``m14.novelty.oof_explained``)
    G17.d  generalisation                held-out split-half: BLUPs from odd vs even sequences (fixed other-fold parameters),
                                         position-residualised Pearson ≥ split_half_min
    G17.e  NFL relevance                 within-position partial Spearman with ≥ 1 eligible M16 target, bootstrap CI excluding 0
                                         after BH over signatures × targets
    G17.f  interpretability (label)      |ρ| ≥ label_min_abs_rho with its a-priori physical anchor (``ANCHORS``)
Status: validated (a–e) · measurement_valid_not_nfl_relevant (a–d) · invalid (a or b FAIL) · exploratory otherwise ·
NOT_EVALUATED when an input is missing. The label is the metric name only when f PASSES; otherwise "unlabelled".
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from ..status import FAIL, NOT_EVALUATED, PASS
from . import hierarchical as HM

NAN = float("nan")
# a-priori physical anchors (Combine agility simple features from M16 ``simple_tracking_table``); sign not imposed
ANCHORS = {"max_controlled_decel": "trk_agility__min_atan", "braking_point_efficiency": "trk_agility__min_atan",
           "braking_budget_utilization": "trk_agility__braking_frac", "stop_go_elasticity": "trk_agility__accel_frac",
           "turn_speed_retention": "trk_agility__max_abs_an", "corner_exit_efficiency": "trk_agility__q90_speed",
           "brake_propulsion_transition": "trk_agility__jerk_rms", "first_step_explosiveness": "trk_agility__max_atan",
           "cut_recovery_time": "trk_agility__max_atan"}


def signature_table(effects: Any) -> Any:
    """Wide Combine signature table: nfl_id, position_group, fold, sig_<metric> (BLUP), n_<metric> (events)."""
    import polars as pl

    c = effects.filter(pl.col("domain") == "combine")
    if not c.height:
        return pl.DataFrame()
    w = c.pivot(on="metric", index=["nfl_id", "position_group", "fold"], values="blup")
    w = w.rename({m: f"sig_{m}" for m in w.columns if m not in ("nfl_id", "position_group", "fold")})
    n = c.pivot(on="metric", index="nfl_id", values="n_events")
    n = n.rename({m: f"n_{m}" for m in n.columns if m != "nfl_id"})
    return w.join(n, on="nfl_id", how="left").sort("nfl_id")


def split_half_blups(events: Any, metric: str, fold_of: dict[int, int], use_statsmodels: bool, min_athletes: int) -> Any:
    """Held-out split-half: BLUPs computed separately from odd and even sequences (``row`` parity)."""
    import polars as pl

    parts = []
    for h in (0, 1):
        e, _ = HM.athlete_effects(events.filter(pl.col("row") % 2 == h), metric, "combine", fold_of, use_statsmodels, min_athletes)
        if e.height:
            parts.append(e.select("nfl_id", "position_group", pl.col("blup").alias(f"b{h}")))
    if len(parts) < 2:
        return pl.DataFrame()
    return parts[0].join(parts[1].select("nfl_id", "b1"), on="nfl_id")


def _pboot(x: np.ndarray, y: np.ndarray, P: np.ndarray, n_boot: int, seed: int) -> tuple[float, float, float, float]:
    from ..stats import partial_spearman

    r = partial_spearman(x, y, P)
    rng = np.random.default_rng(seed)
    bs = np.array([partial_spearman(x[i], y[i], P[i]) for i in (rng.integers(0, x.size, x.size) for _ in range(n_boot))])
    bs = bs[np.isfinite(bs)]
    if not bs.size:
        return r, NAN, NAN, NAN
    p = min(1.0, 2 * float(min((bs <= 0).mean(), (bs >= 0).mean())))
    return r, float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975)), p


def _dummies(pos: np.ndarray) -> np.ndarray:
    lv = sorted(set(pos.tolist()))[1:]
    return np.stack([(pos == v).astype(float) for v in lv], 1) if lv else np.zeros((pos.size, 0))


def evaluate(effects: Any, summary: Any, components: Any, events: Any, fold_of: dict[int, int], analysis: Any, targets: Any, target_cols: Sequence[str],
             anchors: Any, cfg: Any) -> tuple[Any, Any]:
    """(per signature × criterion gate rows, per signature × target association table)."""
    import polars as pl

    from ..m14.novelty import oof_explained
    from ..m16.target_reliability import _pos_resid
    from ..stats import bh
    from ...data.validation import M1_COLS

    sig = signature_table(effects)
    if not sig.height:
        return pl.DataFrame(), pl.DataFrame()
    metrics = [c.removeprefix("sig_") for c in sig.columns if c.startswith("sig_")]
    comb = components.filter(pl.col("domain") == "combine") if components is not None and components.height else None
    assoc = []
    for k, m in enumerate(metrics):
        col = f"sig_{m}"
        j = sig.select("nfl_id", "position_group", col).join(targets.select("nfl_id", *[c for c in target_cols if c in targets.columns]), on="nfl_id", how="inner")
        for t in target_cols:
            if t not in j.columns:
                continue
            d = j.select("position_group", col, t).drop_nulls().filter(pl.col(col).is_finite() & pl.col(t).cast(pl.Float64).is_finite())
            if d.height < 30:
                continue
            pos = d["position_group"].fill_null("?").to_numpy()
            r, lo, hi, p = _pboot(d[col].to_numpy(), d[t].cast(pl.Float64).to_numpy(), _dummies(pos), cfg.n_boot // 2, cfg.seed + 7 * k)
            assoc.append({"metric": m, "target_column": t, "n": d.height, "partial_spearman": r, "lo": lo, "hi": hi, "p_boot": p})
    assoc_t = pl.DataFrame(assoc, infer_schema_length=None)
    if assoc_t.height:
        assoc_t = assoc_t.with_columns(pl.Series("p_bh", bh([1.0 if not np.isfinite(v) else v for v in assoc_t["p_boot"].to_list()])))
    rows = []
    frame = analysis.join(sig.drop("position_group", "fold"), on="nfl_id", how="left") if analysis is not None else None
    for k, m in enumerate(metrics):
        col = f"sig_{m}"
        lab = {"signature": m}
        cm = comb.filter(pl.col("metric") == m) if comb is not None else None
        if cm is not None and cm.height:
            share_r = (cm["var_u"] / (cm["var_u"] + cm["var_e"])).to_numpy()
            share_t = (cm["var_u"] / cm["var_total"]).to_numpy()
            a_v, a_min = float(np.nanmedian(share_r)), float(np.nanmin(share_r))
            b_v, b_min = float(np.nanmedian(share_t)), float(np.nanmin(share_t))
            ra = {"criterion": "G17.a repeatability after context", "value": a_v, "ci_lo": a_min, "threshold": cfg.icc_hi,
                  "status": PASS if (a_v >= cfg.icc_hi and a_min >= cfg.icc_lo) else FAIL}
            rb = {"criterion": "G17.b athlete share of total", "value": b_v, "ci_lo": b_min, "threshold": cfg.athlete_share_min,
                  "status": PASS if (b_v >= cfg.athlete_share_min and b_min >= 0.25) else FAIL}
        else:
            ra = {"criterion": "G17.a repeatability after context", "status": NOT_EVALUATED}
            rb = {"criterion": "G17.b athlete share of total", "status": NOT_EVALUATED}
        if frame is not None and col in frame.columns:
            nv = oof_explained(frame, col, [c for c in M1_COLS if c in frame.columns], True)
            v = nv.get("oof_r2")
            rc = {"criterion": "G17.c novelty", "value": v, "threshold": cfg.novelty_max_r2,
                  "status": (PASS if v <= cfg.novelty_max_r2 else FAIL) if v is not None and np.isfinite(v) else NOT_EVALUATED}
        else:
            rc = {"criterion": "G17.c novelty", "status": NOT_EVALUATED}
        sh = split_half_blups(events, m, fold_of, cfg.use_statsmodels, cfg.min_reliability_athletes)
        if sh.height >= 30:
            pos = sh["position_group"].fill_null("?").to_numpy()
            a, b = _pos_resid(sh["b0"].to_numpy(), pos), _pos_resid(sh["b1"].to_numpy(), pos)
            ok = np.isfinite(a) & np.isfinite(b)
            r = float(np.corrcoef(a[ok], b[ok])[0, 1]) if ok.sum() >= 30 else NAN
            rd = {"criterion": "G17.d held-out split-half", "value": r, "threshold": cfg.split_half_min,
                  "status": (PASS if r >= cfg.split_half_min else FAIL) if np.isfinite(r) else NOT_EVALUATED}
        else:
            rd = {"criterion": "G17.d held-out split-half", "status": NOT_EVALUATED}
        a_m = assoc_t.filter(pl.col("metric") == m) if assoc_t.height else assoc_t
        if a_m is not None and a_m.height:
            sig_rel = a_m.filter(((pl.col("lo") > 0) | (pl.col("hi") < 0)) & (pl.col("p_bh") <= 0.05))
            re_ = {"criterion": "G17.e NFL relevance", "value": float(a_m["partial_spearman"].abs().max()), "threshold": None,
                   "status": PASS if sig_rel.height else FAIL, "note": ", ".join(sig_rel["target_column"].to_list())}
        else:
            re_ = {"criterion": "G17.e NFL relevance", "status": NOT_EVALUATED, "note": "no eligible M16 target with ≥ 30 athletes"}
        anc = ANCHORS.get(m)
        if anchors is not None and anc in getattr(anchors, "columns", []) and col in sig.columns:
            d = sig.select("nfl_id", col).join(anchors.select("nfl_id", anc), on="nfl_id").drop_nulls()
            from ..stats import spearman

            rho = spearman(d[col].to_numpy().astype(float), d[anc].to_numpy().astype(float)) if d.height >= 30 else NAN
            rf = {"criterion": "G17.f interpretability (label)", "value": rho, "threshold": cfg.label_min_abs_rho,
                  "status": (PASS if abs(rho) >= cfg.label_min_abs_rho else FAIL) if np.isfinite(rho) else NOT_EVALUATED, "note": f"anchor {anc}"}
        else:
            rf = {"criterion": "G17.f interpretability (label)", "status": NOT_EVALUATED, "note": f"anchor {anc} unavailable"}
        crit = [ra, rb, rc, rd, re_, rf]
        st = {r["criterion"][:5]: r["status"] for r in crit}
        if st["G17.a"] == FAIL or st["G17.b"] == FAIL:
            status = "invalid"
        elif any(st[x] == NOT_EVALUATED for x in ("G17.a", "G17.b", "G17.c", "G17.d")):
            status = NOT_EVALUATED
        elif all(st[x] == PASS for x in ("G17.a", "G17.b", "G17.c", "G17.d", "G17.e")):
            status = "validated"
        elif all(st[x] == PASS for x in ("G17.a", "G17.b", "G17.c", "G17.d")):
            status = "measurement_valid_not_nfl_relevant"
        else:
            status = "exploratory"
        label = m if st["G17.f"] == PASS else "unlabelled"
        rows += [{**lab, **r} for r in crit]
        rows.append({**lab, "criterion": "G17 decision", "status": status, "note": f"label: {label}"})
    return pl.DataFrame(rows, infer_schema_length=None), assoc_t


def validated_columns(gate: Any) -> list[str]:
    import polars as pl

    if gate is None or not gate.height:
        return []
    t = gate.filter((pl.col("criterion") == "G17 decision") & (pl.col("status") == "validated"))
    return [f"sig_{m}" for m in t["signature"].to_list()]


__all__ = ["ANCHORS", "evaluate", "signature_table", "split_half_blups", "validated_columns"]
