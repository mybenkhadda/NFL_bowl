"""Notebook-level orchestration of M15 Tracks B–E and G15 (T07 / T08). Each ``run_*`` is one guarded notebook step:
it reuses cached outputs where their provenance matches, writes every table through ``M15Run.table`` and returns the
in-memory results. No function here invents a value: a missing input raises ``ResearchBlocked`` with its name.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from ..status import BLOCKED_MISSING_INPUT, ResearchBlocked
from . import gates as G
from . import latent as LT
from . import preservation as PR
from . import training as TR
from . import transfer as TF15


def finished_specs(cfg: Any, run: Any) -> list[TR.FitSpec]:
    return [s for s in TR.grid(cfg) if TR.emb_path(run, s).is_file()]


# ====================================================================== B
def run_b(data: Any, cfg: Any, run: Any, ctx: dict[str, Any], device: str) -> Any:
    import polars as pl

    from . import models as MD

    plan = []
    for a in cfg.architectures:
        for d in cfg.latent_dims:
            if a == "pca":
                plan.append({"arch": a, "d": d, "width": None, "params": None, "ratio": None, "fits_band": True})
            elif MD.TORCH:
                plan.append(MD.choose_width(a, d, cfg))
    ctx["outputs"] += run.table(pl.DataFrame(plan, infer_schema_length=None), "parameter_budget", "latent")
    status = TR.run_grid(cfg, data, run, device)
    ctx["outputs"] += run.table(status, "training_status", "latent")
    hist = []
    for s in TR.grid(cfg):
        h = TR.ckpt_dir(run, s) / "history.parquet"
        if h.is_file():
            hist.append(pl.read_parquet(h).with_columns(pl.lit(s.arch).alias("arch"), pl.lit(s.d).alias("d"), pl.lit(s.seed).alias("seed"), pl.lit(s.fold).alias("fold")))
    if hist:
        ctx["outputs"] += run.table(pl.concat(hist, how="diagonal_relaxed"), "training_curves", "latent")
    return status


# ====================================================================== C
def run_c(data: Any, cfg: Any, run: Any, ctx: dict[str, Any], device: str, n_per_domain: int = 400) -> dict[str, Any]:
    import polars as pl

    specs = finished_specs(cfg, run)
    if not specs:
        raise ResearchBlocked(BLOCKED_MISSING_INPUT, "no fitted encoder with exported embeddings (run section B)", missing=["embeddings/inductive/*"])
    parts, robust = [], []
    nfl_len = data.index.filter(pl.col("domain") == "nfl")["n_valid"].to_numpy()
    rob_fold = cfg.selected_folds()[0]
    for s in specs:
        rows = PR.sample_rows(data, s.fold, n_per_domain, cfg.seed, cfg.min_segment_frames)
        parts.append(PR.preservation_for_spec(s, data, cfg, run, rows, device))
        if s.fold == rob_fold:
            robust.append(PR.robustness(s, data, cfg, run, rows, nfl_len, device, cfg.seed))
    tab = pl.concat(parts, how="diagonal_relaxed")
    run.data_file(tab, "predictions/reconstruction/preservation_rows", {"bundle_fingerprint": data.fingerprint, "config_fingerprint": cfg.fingerprint()})
    summ = PR.summary(tab)
    paired = PR.paired_vs_reference(tab, cfg.n_boot_screen, cfg.seed) if "pca" in cfg.architectures else pl.DataFrame()
    rob = pl.concat(robust, how="diagonal_relaxed") if robust else pl.DataFrame()
    for name, df in (("metrics", summ), ("paired_vs_pca", paired), ("robustness", rob)):
        ctx["outputs"] += run.table(df, name, "preservation")
    return {"rows": tab, "summary": summ, "paired": paired, "robustness": rob}


# ====================================================================== D
def athlete_feature_table(feat: Any) -> Any:
    """Held-out-agnostic athlete medians of simple forty features (interpretation targets; Combine data only)."""
    import polars as pl

    cols = [c for c in ("max_speed", "q90_speed", "max_atan", "min_atan", "braking_frac", "jerk_rms", "entry_speed") if c in feat.columns]
    return feat.filter((pl.col("domain") == "combine") & (pl.col("kind") == "forty")).group_by("nfl_id").agg([pl.col(c).median().alias(f"forty_{c}") for c in cols])


def run_d(data: Any, cfg: Any, run: Any, ctx: dict[str, Any], feat: Any | None = None, icc_boot: int = 0) -> dict[str, Any]:
    import polars as pl

    specs = finished_specs(cfg, run)
    if not specs:
        raise ResearchBlocked(BLOCKED_MISSING_INPUT, "no exported embeddings (run section B)", missing=["embeddings/inductive/*"])
    geo, prb, icc, rep, interp, proj = [], [], [], [], [], []
    afeat = athlete_feature_table(feat) if feat is not None else None
    cache: dict[tuple, Any] = {}
    for s in specs:
        emb = TR.load_embeddings(run, s)
        cache[(s.arch, s.d, s.seed, s.fold)] = emb
        lab = {"arch": s.arch, "d": s.d, "seed": s.seed, "fold": s.fold}
        geo.append(LT.geometry(emb, lab))
        prb.append(LT.probes(emb, lab))
        icc.append(LT.dim_icc(emb, lab, n_boot=icc_boot, seed=cfg.seed))
        rep.append(LT.repeat_auc(emb, lab))
        if afeat is not None:
            it = LT.interpretation(emb, afeat, lab, seed=cfg.seed)
            if it.height:
                interp.append(it)
        if s.fold == cfg.selected_folds()[0]:
            proj.append(LT.projection(emb, lab, seed=cfg.seed))
    seeds = []
    for (a, d, s1, f), e1 in cache.items():
        for (a2, d2, s2, f2), e2 in cache.items():
            if a2 == a and d2 == d and f2 == f and s2 > s1:
                seeds.append(LT.seed_stability(e1, e2, {"arch": a, "d": d, "seed_a": s1, "seed_b": s2, "fold": f}))
    out = {"geometry": pl.DataFrame(geo, infer_schema_length=None), "probes": pl.DataFrame(prb, infer_schema_length=None),
           "dim_icc": pl.concat(icc, how="diagonal_relaxed") if icc else pl.DataFrame(), "repeat_auc": pl.concat(rep, how="diagonal_relaxed"),
           "seed_stability": pl.DataFrame(seeds, infer_schema_length=None) if seeds else pl.DataFrame(),
           "interpretation": pl.concat(interp, how="diagonal_relaxed") if interp else pl.DataFrame(),
           "projection": pl.concat(proj, how="diagonal_relaxed") if proj else pl.DataFrame()}
    out["icc_summary"] = LT.icc_summary(out["dim_icc"])
    out["labels"] = LT.labels(out["interpretation"]) if out["interpretation"].height else pl.DataFrame()
    for name in ("geometry", "probes", "repeat_auc", "seed_stability", "interpretation", "labels"):
        ctx["outputs"] += run.table(out[name], name, "latent")
    ctx["outputs"] += run.table(out["dim_icc"], "latent_icc", "reliability") + run.table(out["icc_summary"], "latent_icc_summary", "reliability")
    if out["projection"].height:
        run.data_file(out["projection"], "features/projections/latent_projection", {"config_fingerprint": cfg.fingerprint()})
    return out


# ====================================================================== E
def run_e(data: Any, cfg: Any, run: Any, ctx: dict[str, Any], forty: dict[int, float] | None = None, e0: np.ndarray | None = None,
          concat_pairs: Sequence[tuple[str, str]] = (("stft_ae", "lstm_ae"),)) -> dict[str, Any]:
    """Retrieval arms for every fitted representation and fold, plus F0, concatenations, position chance, damage checks."""
    import polars as pl

    specs = finished_specs(cfg, run)
    if not specs:
        raise ResearchBlocked(BLOCKED_MISSING_INPUT, "no exported embeddings (run section B)", missing=["embeddings/inductive/*"])
    rows, chance, damage = [], [], []
    by = {}
    for s in specs:
        emb = TR.load_embeddings(run, s)
        by[(s.arch, s.d, s.seed, s.fold)] = emb
        lab = {"rep": f"{s.arch}_d{s.d}", "arch": s.arch, "d": s.d, "seed": s.seed, "fold": s.fold}
        try:
            r, ex = TF15.retrieval_for_embedding(emb, lab, forty, seed=cfg.seed + s.fold)
        except ResearchBlocked as e:
            chance.append({**lab, "note": e.reason})
            continue
        rows += r
        chance.append({**lab, **ex["position_chance"]})
        for arm, fmap in ex["maps"].items():
            if arm == "infonce":
                damage.append({**TF15.damage_check(emb, {**lab, "arm": arm}, fmap)})
    for a, b in concat_pairs:
        for (arch, d, seed, f), ea in list(by.items()):
            if arch != a or (b, d, seed, f) not in by:
                continue
            emb = TF15.concat_embeddings(ea, by[(b, d, seed, f)])
            lab = {"rep": f"{a}+{b}_d{d}", "arch": f"{a}+{b}", "d": d, "seed": seed, "fold": f}
            try:
                r, _ = TF15.retrieval_for_embedding(emb, lab, forty, arms=("unaligned", "procrustes"), seed=cfg.seed + f)
                rows += r
            except ResearchBlocked:
                pass
    if e0 is not None:
        for f in cfg.selected_folds():
            emb = TF15.sequence_matrix_embedding(e0, data.index, f)
            lab = {"rep": "F0", "arch": "F0", "d": int(e0.shape[1]), "seed": 0, "fold": f}
            try:
                r, _ = TF15.retrieval_for_embedding(emb, lab, forty, arms=("unaligned", "procrustes"), seed=cfg.seed + f)
                rows += r
            except ResearchBlocked:
                pass
    scores = pl.DataFrame(rows, infer_schema_length=None)
    summ = TF15.summarise(scores)
    paired = []
    for rep_ in scores["rep"].unique().to_list():
        sub = scores.filter(pl.col("rep") == rep_)
        for arm in ("procrustes", "infonce", "random"):
            for pool in TF15.POOLS:
                r = TF15.paired_rr(sub, arm, "unaligned", pool, cfg.n_boot_screen, cfg.seed)
                if r.get("n", 0) >= 10:
                    paired.append({"rep": rep_, **r})
    if "F0" in scores["rep"].unique().to_list():
        f0 = scores.filter((pl.col("rep") == "F0") & (pl.col("arm") == "unaligned")).select("fold", "nfl_id", "pool", pl.col("rr").alias("rr_f0"))
        for rep_ in [r for r in scores["rep"].unique().to_list() if r != "F0"]:
            for pool in TF15.POOLS:
                j = scores.filter((pl.col("rep") == rep_) & (pl.col("arm") == "unaligned") & (pl.col("pool") == pool)).join(f0, on=["fold", "nfl_id", "pool"])
                if j.height >= 10:
                    d = (j["rr"] - j["rr_f0"]).to_numpy()
                    bs = d[np.random.default_rng(cfg.seed).integers(0, d.size, (cfg.n_boot_screen, d.size))].mean(1)
                    paired.append({"rep": rep_, "arm": "unaligned", "reference": "F0", "pool": pool, "n": int(d.size), "delta_mrr": float(d.mean()),
                                   "lo": float(np.quantile(bs, 0.025)), "hi": float(np.quantile(bs, 0.975)), "p_boot_le0": float((bs <= 0).mean())})
    out = {"scores": scores, "summary": summ, "paired": pl.DataFrame(paired, infer_schema_length=None), "position_chance": pl.DataFrame(chance, infer_schema_length=None),
           "damage": pl.DataFrame(damage, infer_schema_length=None), "prototype": pl.DataFrame([TF15.prototype_reading()])}
    run.data_file(scores, "datasets/cached/retrieval_scores", {"config_fingerprint": cfg.fingerprint()})
    for name in ("summary", "paired", "position_chance", "damage", "prototype"):
        ctx["outputs"] += run.table(out[name], name, "retrieval")
    return out


# ====================================================================== G15
def run_gate(cfg: Any, run: Any, ctx: dict[str, Any]) -> Any:
    def frame(name: str, fam: str) -> Any:
        return run.frame(name, fam) if run.has(name, fam) else None

    gate = G.g15(frame("paired_vs_pca", "preservation"), frame("metrics", "preservation"), frame("latent_icc_summary", "reliability"),
                 frame("geometry", "latent"), frame("seed_stability", "latent"), frame("robustness", "preservation"), list(cfg.architectures),
                 [int(d) for d in cfg.latent_dims])
    ctx["outputs"] += run.table(gate, "m15_gate", "synthesis")
    return gate


__all__ = ["athlete_feature_table", "finished_specs", "run_b", "run_c", "run_d", "run_e", "run_gate"]
