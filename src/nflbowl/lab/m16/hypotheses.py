"""E16-C1 / C2 / C3 — pre-declared drill-to-NFL research hypotheses [HYP] (not established transfer relationships).

    E16-C1  forty → NFL high-speed movement          targets T1 (and T0)          drill scope ``forty``
    E16-C2  short shuttle → NFL braking / reacceleration   T3, T4                 drill scope ``short_shuttle``
    E16-C3  three-cone → NFL redirection             T5 and the context-matched ``t16_ctx_three_cone__T5``   scope ``three_cone``

Per hypothesis and target, one ladder on the official engine: M1 · M1 + drill PCA · M1 + drill STFT · M1 + drill LSTM ·
M1 + drill handcrafted kinematics (the acceleration-profile control: ``kin_<drill>__*``; for C1 also M1+F0, the engineered
forty statistics). Embedding blocks are the drill-scope aggregates of the VERIFIED_FOLD_SAFE representations (primary
aggregation). Holm over the hypothesis family (each drill model vs M1).

Prerequisites (checked before any fit; a failed one is recorded, never bypassed):
* the target exists with ≥ ``hypothesis_min_athletes`` athletes with M1 and the drill  → else BLOCKED_INSUFFICIENT_SAMPLE
* E16-C2 / C3: the target's G16-A status is not FAIL (measurement validity) → else NOT_RUN with the reason
* E16-C2: censored / incomplete share ≤ ``max_censored_share``                 → else NOT_RUN
* E16-C3 context-matched outcome: the three-cone common support is COMPLETED   → else BLOCKED_INSUFFICIENT_OVERLAP
"""

from __future__ import annotations

from typing import Any, Sequence

from ..status import BLOCKED_INSUFFICIENT_OVERLAP, BLOCKED_INSUFFICIENT_SAMPLE, COMPLETED, FAIL, NOT_RUN

HYPOTHESES = {
    "E16-C1": {"drill": "forty", "kin_scope": "forty", "targets": ("T1", "T0"), "validity_required": False, "censoring": None,
               "question": "Does the forty-yard movement signature add to M1 for NFL high-speed capacity?"},
    "E16-C2": {"drill": "short_shuttle", "kin_scope": "short_shuttle", "targets": ("T3", "T4"), "validity_required": True,
               "censoring": {"T3": "share_incomplete", "T4": "share_propulsion_censored"},
               "question": "Does short-shuttle braking / reacceleration add to M1 for NFL braking and reacceleration?"},
    "E16-C3": {"drill": "three_cone", "kin_scope": "three_cone", "targets": ("T5", "ctx:t16_ctx_three_cone__T5"), "validity_required": True, "censoring": None,
               "question": "Does three-cone redirection add to M1 for NFL redirection at comparable turning demand?"},
}


def target_column(t: str) -> tuple[str, str]:
    """(target id label, analysis column) — ``ctx:<col>`` names a context-matched outcome column."""
    from .targets import TARGETS

    if t.startswith("ctx:"):
        return t.removeprefix("ctx:"), t.removeprefix("ctx:")
    return t, TARGETS[t].column


def prerequisites(hid: str, t: str, analysis: Any, gate_a: Any, ath: Any, support: Any, cfg: Any) -> tuple[str, str]:
    """(status, reason) of one hypothesis × target before fitting."""
    import polars as pl

    from ...data.validation import M1_COLS

    h = HYPOTHESES[hid]
    tid, col = target_column(t)
    if t.startswith("ctx:"):
        st = support.filter(pl.col("drill") == h["drill"])["status"].to_list() if support is not None and support.height else []
        if not st or st[0] != COMPLETED:
            return BLOCKED_INSUFFICIENT_OVERLAP, f"{h['drill']} vs NFL common support: {st[0] if st else 'not computed'}"
    if col not in analysis.columns:
        return BLOCKED_INSUFFICIENT_SAMPLE, f"{col} absent from the analysis frame"
    kin = [c for c in analysis.columns if c.startswith(f"kin_{h['kin_scope']}__") and not c.endswith("__n_attempts")]
    has_drill = pl.any_horizontal([pl.col(c).is_not_null() for c in kin]) if kin else pl.lit(False)
    n = analysis.filter(pl.col(col).cast(pl.Float64).is_finite() & pl.all_horizontal([pl.col(c).is_not_null() for c in M1_COLS if c in analysis.columns])
                        & has_drill).height
    if n < cfg.hypothesis_min_athletes:
        return BLOCKED_INSUFFICIENT_SAMPLE, f"{n} athletes with {col}, M1 and {h['drill']} tracking (< {cfg.hypothesis_min_athletes})"
    if h["validity_required"] and not t.startswith("ctx:") and gate_a is not None and gate_a.height:
        g = gate_a.filter(pl.col("target_id") == tid)["status"].to_list()
        if g and g[0] == FAIL:
            return NOT_RUN, f"measurement validity: {tid} G16-A FAIL ({gate_a.filter(pl.col('target_id') == tid)['reason'][0]})"
    cens = (h["censoring"] or {}).get(tid)
    if cens and ath is not None and cens in ath.columns:
        share = float(ath[cens].drop_nulls().median() or 0.0)
        if share > cfg.max_censored_share:
            return NOT_RUN, f"censoring: median {cens} {share:.2f} > {cfg.max_censored_share}"
    return COMPLETED, f"{n} athletes eligible"


def drill_models(cfg: Any, plan: dict[str, Any], analysis: Any, fold_of: dict[int, int], outer_folds: int, hid: str, log: list) -> tuple[dict[str, Any], Any]:
    """(official-engine models of one hypothesis, feature inventory)."""
    import polars as pl

    from ... import _backend
    from ..m14.transfer import coverage_factory
    from . import features as FT
    from .baselines import assert_no_nfl_predictors
    from .ladder import ARCH

    ae = _backend.ae()
    h = HYPOTHESES[hid]
    m1 = list(ae.M1_COLS)
    mods: dict[str, Any] = {"M1": ae.Model(m1)}
    inv = []
    for r, arch in ARCH.items():
        if plan[r].status != "eligible":
            continue
        t, i = FT.by_fold(cfg.m15_run_dir, plan[r].rep, h["drill"], cfg.primary_aggregation, fold_of, outer_folds, cfg.trim_share)
        inv.append(i.with_columns(pl.lit(hid).alias("hypothesis")))
        mods[f"M1+{h['drill']}_{arch}"] = ae.Model(m1, factory=FT.block_factory(fold_of, t, cfg.embedding_k))
    kin = FT.kinematic_columns(analysis, h["kin_scope"])
    if kin:
        assert_no_nfl_predictors(kin)
        mods[f"M1+{h['drill']}_kinematics"] = ae.Model(m1, factory=coverage_factory(analysis, kin, cfg.min_coverage, log, f"{hid} kinematics"))
    eng = [c for c in analysis.columns if c.startswith("eng")]
    if hid == "E16-C1" and eng:
        mods["M1+F0"] = ae.Model([*m1, *eng])
    return mods, (pl.concat(inv, how="diagonal_relaxed") if inv else pl.DataFrame())


def family(mods: dict[str, Any]) -> list[tuple[str, str]]:
    return [(m, "M1") for m in mods if m != "M1"]


__all__ = ["HYPOTHESES", "drill_models", "family", "prerequisites", "target_column"]
