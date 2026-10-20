"""Notebook-level orchestration of M17 (T05, T12–T14). Upstream runs (M14, M15, M16) are READ only; each ``run_*`` is one
guarded step returning its in-memory results and writing every table through ``M17Run.table``."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from ..status import BLOCKED_MISSING_INPUT, ResearchBlocked
from . import alignment as AL
from . import censoring as CS
from . import contexts as CX
from . import events as EV
from . import fusion as FU
from . import hierarchical as HM
from . import signatures as SG


def load_parquet(path: Path) -> Any:
    import polars as pl

    return pl.read_parquet(path) if path.is_file() else None


# ====================================================================== A1–A3
def run_a1(cfg: Any, data: Any, run: Any, ctx: dict[str, Any]) -> dict[str, Any]:
    counts, grid_icc, evs = EV.threshold_grid(data, run, cfg)
    default = evs["default"]
    fm = EV.failure_modes(default)
    icc = EV.per_drill_icc(default, cfg.n_boot // 2, cfg.seed, cfg.min_reliability_athletes)
    diag = EV.diagnosis(fm, icc, grid_icc)
    for name, df in (("failure_modes", fm), ("per_drill_icc", icc), ("threshold_grid_counts", counts), ("threshold_grid_icc", grid_icc), ("diagnosis", diag)):
        ctx["outputs"] += run.table(df, name, "events_v2")
    return {"events": default, "failure_modes": fm, "icc": icc, "grid_icc": grid_icc, "diagnosis": diag}


def all_events(cfg: Any, combine_events: Any) -> tuple[Any, str]:
    """Combine events of the default variant + every NFL event (M16 cache, read only) when available."""
    import polars as pl

    nfl = load_parquet(Path(cfg.m16_run_dir) / "features" / "targets" / "nfl_events.parquet")
    c = combine_events.filter(pl.col("domain") == "combine")
    if nfl is not None:
        return pl.concat([c, nfl], how="diagonal_relaxed"), "Combine: default variant (all rows); NFL: M16 full NFL extraction"
    return combine_events, f"NFL: seeded sample of {cfg.grid_nfl_sample} windows (M16 NFL events not found)"


def run_a2(cfg: Any, events: Any, run: Any, ctx: dict[str, Any]) -> Any:
    st = CX.assign_strata(events, cfg)
    cov = CX.coverage(st, cfg)
    run.data_file(st, "features/events_v2/events_with_strata", {"config_fingerprint": cfg.fingerprint()})
    ctx["outputs"] += run.table(cov, "stratum_coverage", "context")
    return {"strata": st, "coverage": cov}


def run_a3(cfg: Any, strata: Any, run: Any, ctx: dict[str, Any]) -> dict[str, Any]:
    rec = CS.recovery_records(strata)
    if not rec.height:
        raise ResearchBlocked(BLOCKED_MISSING_INPUT, "events lack recovery columns", missing=["cut_recovery_time", "recovery_censored"])
    ath = CS.athlete_recovery(rec)
    rel = CS.censoring_reliability(rec, cfg.n_boot // 2, cfg.seed)
    ctx["outputs"] += run.table(ath, "recovery_by_athlete", "events_v2") + run.table(rel, "censoring_reliability", "events_v2")
    return {"athletes": ath, "reliability": rel}


# ====================================================================== B
def run_b(cfg: Any, strata: Any, fold_of: dict[int, int], run: Any, ctx: dict[str, Any]) -> dict[str, Any]:
    out = HM.run_all(strata, EV.KEY_METRICS, fold_of, cfg.use_statsmodels, cfg.min_reliability_athletes, cfg.n_boot // 2, cfg.seed)
    run.data_file(out["effects"], "features/events_v2/athlete_effects", {"config_fingerprint": cfg.fingerprint()})
    for name in ("components", "summary", "cross_domain"):
        ctx["outputs"] += run.table(out[name], name, "variance")
    return out


# ====================================================================== C / D
def eligible(cfg: Any) -> tuple[list[tuple[str, int]], str]:
    """M15-gate-eligible representations that also passed the M16 leakage audit (when that audit exists)."""
    import polars as pl

    from ..m16.retrieval import eligible_representations

    reps, note = eligible_representations(cfg.m15_run_dir, cfg.allow_descriptive)
    v = load_parquet(Path(cfg.m16_run_dir) / "tables" / "retrieval" / "leakage_verdict.parquet")
    if v is not None and v.height:
        ok = set(v.filter(pl.col("verdict") == "combine_only_inductive")["rep"].to_list())
        reps = [r for r in reps if f"{r[0]}_d{r[1]}" in ok]
        note += f"; {len(reps)} after the M16 leakage audit"
    return reps, note


def run_c(cfg: Any, athletes: Any, folds: list[int], run: Any, ctx: dict[str, Any]) -> dict[str, Any]:
    reps, note = eligible(cfg)
    if len(reps) < 2:
        raise ResearchBlocked(BLOCKED_MISSING_INPUT, f"fusion needs two eligible representations ({note})", missing=["G15 GO representations"])
    pair = (reps[0], reps[1])
    stft = [r for r in reps if r[0] == "stft_ae"]
    rnn = [r for r in reps if r[0] in ("lstm_ae", "gru_ae")]
    if stft and rnn:
        pair = (stft[0], rnn[0])
    scores, written = FU.run_fusion(cfg.m15_run_dir, pair[0], pair[1], athletes, run, folds, cfg.seed)
    adopt = FU.adoption(scores)
    ctx["outputs"] += run.table(scores, "fusion_scores", "fusion") + run.table(adopt, "fusion_adoption", "fusion")
    return {"pair": pair, "scores": scores, "adoption": adopt, "written": {k: str(v) for k, v in written.items()}}


def sequence_strata(cfg: Any, feat: Any) -> Any:
    """Per sequence entry-speed stratum (same edges as the event strata) for context-matched pairs."""
    import polars as pl

    edges = np.asarray(cfg.entry_speed_bins)
    labels = CX._bin_labels(cfg.entry_speed_bins, "yd/s")
    v = feat["entry_speed"].cast(pl.Float64).fill_null(np.nan).to_numpy()
    lab = [labels[i] if np.isfinite(x) else "entry_unknown" for i, x in zip(np.searchsorted(edges, v, side="right"), v, strict=True)]
    return feat.select("idx").with_columns(pl.Series("s_entry", lab))


def run_d(cfg: Any, athletes: Any, feat: Any, folds: list[int], run: Any, ctx: dict[str, Any]) -> dict[str, Any]:
    import polars as pl

    from ..m16.retrieval import embedding_files, forty_map

    reps, note = eligible(cfg)
    if not reps:
        raise ResearchBlocked(BLOCKED_MISSING_INPUT, f"no eligible representation ({note})", missing=["G15 GO representations"])
    strata = sequence_strata(cfg, feat)
    forty = forty_map(athletes)
    rows, dmg = [], []
    for a, d in reps:
        for f, p in embedding_files(cfg.m15_run_dir, a, d).items():
            if f not in folds:
                continue
            r, dm = AL.run_alignment(pl.read_parquet(p), strata, {"rep": f"{a}_d{d}", "fold": f}, forty, cfg.seed + f, cfg.training_enabled())
            rows += r
            dmg += dm
    summary = pl.DataFrame(rows, infer_schema_length=None)
    damage = pl.DataFrame(dmg, infer_schema_length=None) if dmg else pl.DataFrame()
    adopt = AL.adoption(summary, damage)
    for name, df in (("alignment_scores", summary), ("alignment_damage", damage), ("alignment_adoption", adopt)):
        ctx["outputs"] += run.table(df, name, "alignment")
    return {"summary": summary, "damage": damage, "adoption": adopt}


# ====================================================================== E + G17
def run_e(cfg: Any, B: dict[str, Any], strata: Any, fold_of: dict[int, int], run: Any, ctx: dict[str, Any]) -> dict[str, Any]:
    from ..m16.targets import TARGETS

    m16 = Path(cfg.m16_run_dir)
    analysis = load_parquet(m16 / "datasets" / "cached" / "analysis_frame.parquet")
    targets = load_parquet(m16 / "features" / "targets" / "athlete_targets.parquet")
    frz = m16 / "manifests" / "target_freeze_m16.json"
    eligible_t = json.loads(frz.read_text(encoding="utf-8")).get("eligible_task_targets", []) if frz.is_file() else []
    tcols = [TARGETS[t].column for t in ["T0", *eligible_t] if t in TARGETS]
    if targets is None:
        raise ResearchBlocked(BLOCKED_MISSING_INPUT, "M16 athlete targets not found (run M16 E16.A2)", missing=[str(m16 / "features/targets/athlete_targets.parquet")])
    if analysis is not None and "nfl.q90_shared" in analysis.columns and "nfl.q90_shared" not in targets.columns:
        targets = targets.join(analysis.select("nfl_id", "nfl.q90_shared"), on="nfl_id", how="left")
    gate, assoc = SG.evaluate(B["effects"], B["summary"], B["components"], strata, fold_of, analysis, targets, tcols, analysis, cfg)
    sig = SG.signature_table(B["effects"])
    run.data_file(sig, "features/events_v2/athlete_signatures", {"config_fingerprint": cfg.fingerprint()})
    ctx["outputs"] += run.table(gate, "m17_gate", "synthesis") + run.table(assoc, "signature_target_association", "signatures")
    ctx["outputs"] += run.table(sig, "athlete_signatures", "signatures")
    return {"gate": gate, "association": assoc, "signatures": sig, "validated": SG.validated_columns(gate)}


__all__ = ["all_events", "eligible", "run_a1", "run_a2", "run_a3", "run_b", "run_c", "run_d", "run_e", "sequence_strata"]
