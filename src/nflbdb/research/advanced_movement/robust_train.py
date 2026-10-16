"""Training-side helpers of notebook 04B (seed replicates, untrained control, leave-one-cohort-out). Needs PyTorch.

Nothing here changes the encoder: architecture, optimiser, budget, input, folds and augmentations are those of
``tcn`` and of the executed notebook 03. The only things that vary are stated in the function names.

Two seeds are kept apart on purpose:
    train_seed   initial weights, batch order, masks and input noise during training, early-stopping mask
    eval_seed    the fixed held-out masking task of the reconstruction tables — the seed of notebook 03 for every
                 replicate, so every seed is scored on the same hidden points
Embeddings do not depend on ``eval_seed`` (they are computed with every recorded point visible).
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, Sequence

import polars as pl
import torch

from ...data.utils import get_logger
from ..sprint_rescue.stats import seed_for
from . import fpca as fp
from . import repr_eval as rv
from . import tcn
from .config import NotebookControls
from .data import fold_athletes, inner_athletes
from .fpca import CurveSet, FpcaConfig
from .fpca_eval import fit_key, fold_tag
from .repr_eval import EvalConfig, MethodSpec
from .tcn import TcnConfig, Trained

log = get_logger(__name__)


def executed_config(models_dir: Path, method: str, repeat: int = 0, fold: int = 0) -> TcnConfig | None:
    """Configuration stored in a checkpoint of notebook 03 (``models_dir``: the upstream model directory), or None
    when the checkpoint was not saved. This is the exact configuration that produced the executed embeddings."""
    p = tcn.checkpoint_path(models_dir, method, repeat, fold)
    return tcn.load_checkpoint(p).tcfg if p.is_file() else None


def reconstructed_config(design_sweep: pl.DataFrame, controls: NotebookControls, fcfg: FpcaConfig) -> TcnConfig:
    """Fallback when no checkpoint exists: the TCN-B configuration rebuilt from the saved design sweep (selected
    consistency term) and the notebook controls. Equal to notebook 03 only if BATCH_SIZE and MAX_EPOCHS are the same."""
    sel = design_sweep.filter(pl.col("selected"))
    if sel.height != 1:
        raise tcn.TcnError(f"the design sweep of notebook 03 marks {sel.height} selected row(s); expected exactly one")
    return TcnConfig(objective=tcn.CONSISTENCY, consistency=sel["consistency"][0], consistency_weight=float(sel["consistency_weight"][0]),
                     min_coverage=fcfg.min_coverage, min_observed_fraction=fcfg.min_observed_fraction).for_mode(controls)


def _folds(splits: pl.DataFrame, repeat: int) -> list[int]:
    return sorted(splits.filter(pl.col("repeat") == repeat)["outer_fold"].unique().to_list())


def fit_seeded(curves: dict[str, CurveSet], splits: pl.DataFrame, plan: Sequence[tuple[MethodSpec, TcnConfig]],
               repeats: Sequence[int], ecfg: EvalConfig, train_seed: int, eval_seed: int, device: str = "cpu",
               supports: dict[tuple[str, int, int], Any] | None = None, models_dir: Path | None = None
               ) -> tuple[dict[str, pl.DataFrame], dict[tuple[str, int, int], Trained]]:
    """``tcn.fit_methods`` with the training seed separated from the evaluation seed (see the module docstring)."""
    emb, rec, hist, fits = [], [], [], []
    models: dict[tuple[str, int, int], Trained] = {}
    for spec, tcfg in plan:
        cs = curves[spec.representation]
        for r in repeats:
            for f in _folds(splits, r):
                t = tcn.train_fold(cs, splits, r, f, spec, tcfg, train_seed, device)
                e, q, counts = tcn.tcn_fold_outputs(t, cs, splits, (supports or {}).get((spec.representation, int(r), int(f))),
                                                    ecfg, eval_seed, device)
                t.info.update(counts)
                key = {"method": spec.method, "repeat": int(r), "outer_fold": int(f)}
                emb.append(e)
                rec.append(q)
                hist += [{**key, **h} for h in t.history]
                fits.append({**key, "representation": spec.representation, "objective": tcfg.objective,
                             "embedding_dim": tcfg.embedding_dim, **t.info})
                models[(spec.method, int(r), int(f))] = t
                if models_dir is not None:
                    tcn.save_checkpoint(tcn.checkpoint_path(models_dir, spec.method, r, f), t)
                log.info("[advanced_movement:ATLAS] %s %s (training seed %d): %d epochs (best %d), %.1f s", spec.method, fold_tag(r, f),
                         train_seed, t.info["epochs_run"], t.info["best_epoch"], t.info["train_seconds"])
    return ({rv.EMBEDDINGS.name: pl.concat(emb), rv.RECON_SEQ.name: pl.concat(rec), tcn.HISTORY.name: rv._table(hist, tcn.HISTORY),
             tcn.FOLDS.name: rv._table(fits, tcn.FOLDS)}, models)


def run_seed_stage(store: Any, stage: str, prefix: str, curves: dict[str, CurveSet], splits: pl.DataFrame,
                   plan: Sequence[tuple[MethodSpec, TcnConfig]], repeats: Sequence[int], ecfg: EvalConfig, train_seed: int,
                   eval_seed: int, device: str, supports: dict[tuple[str, int, int], Any] | None, params: dict[str, Any],
                   files: Sequence[Path], models: dict[tuple[str, int, int], Trained]) -> dict[str, pl.DataFrame]:
    """One cached seed replicate (``store``: ``AtlasStore``, so tables and checkpoints go to the atlas area and can
    never replace those of notebook 03). Restored from cache when parameters and inputs are unchanged; fold models
    are then reloaded from their checkpoints. Tables come back under the base contract names."""
    if not plan:
        raise tcn.TcnError(f"stage {stage}: nothing to train")
    full = {**params, "train_seed": int(train_seed), "eval_seed": int(eval_seed),
            "plan": [{"method": s.method, "representation": s.representation, **t.params()} for s, t in plan]}
    key = fit_key(full, files)
    save = bool(store.controls.save_models)

    def builder() -> dict[str, pl.DataFrame]:
        tables, fitted = fit_seeded(curves, splits, plan, repeats, ecfg, train_seed, eval_seed, device, supports,
                                    store.models_dir if save else None)
        models.update(fitted)
        if save:
            tcn.record_models(store.models_dir, stage, key, [s.method for s, _ in plan])
        return tcn.stage_tables(prefix, tables)
    out = store.build(stage, tcn.stage_contracts(prefix), builder, params=full, files=files)
    if not any(k[0] == plan[0][0].method for k in models):
        models.update(tcn.load_models(store.models_dir, stage, key, plan, splits, repeats))
    return tcn.unstage(prefix, out)


def untrained_outputs(curves: dict[str, CurveSet], splits: pl.DataFrame, spec: MethodSpec, tcfg: TcnConfig, repeats: Sequence[int],
                      ecfg: EvalConfig, seed: int, eval_seed: int, device: str = "cpu",
                      supports: dict[tuple[str, int, int], Any] | None = None) -> dict[str, pl.DataFrame]:
    """Null control: the SAME architecture with its random initial weights, never trained, embedded through the same
    fold-specific normaliser and the same code path as a trained fold model. What it scores is what the input
    representation, the pooling and the normalisation give for free."""
    cs = curves[spec.representation]
    emb, rec = [], []
    for r in repeats:
        for f in _folds(splits, r):
            train, _ = fold_athletes(splits, r, f)
            fit, dev = inner_athletes(splits, r, f, tcfg.inner_dev_fold)
            norm = tcn.CurveNormalizer.fit(cs, train, tcfg.min_coverage, tcfg.min_observed_fraction)
            torch.manual_seed(seed_for(seed, "untrained", spec.method, r, f))
            t = Trained(tcn.CurveTCN(tcfg, cs.grid.size).eval(), norm, tcfg, tuple(sorted(int(i) for i in fit)),
                        tuple(sorted(int(i) for i in dev)), [], {}, spec.method, spec.representation, int(r), int(f))
            e, q, _ = tcn.tcn_fold_outputs(t, cs, splits, (supports or {}).get((spec.representation, int(r), int(f))), ecfg,
                                           eval_seed, device)
            emb.append(e)
            rec.append(q)
    return {rv.EMBEDDINGS.name: pl.concat(emb), rv.RECON_SEQ.name: pl.concat(rec)}


def loco_outputs(curves: dict[str, CurveSet], splits_loco: pl.DataFrame, plan: Sequence[tuple[MethodSpec, TcnConfig]],
                 fpca_spec: MethodSpec | None, fcfg: FpcaConfig, ecfg: EvalConfig, train_seed: int, eval_seed: int,
                 device: str = "cpu", min_heldout: int = 12) -> pl.DataFrame:
    """Leave-one-cohort-out embeddings (full view): for every fold of ``splits_loco`` (``robust.cohort_splits``: fold
    = draft cohort) the encoders of ``plan`` and the FPCA of ``fpca_spec`` are fitted on the OTHER cohorts and applied
    to the held-out one. A cohort too small to hold out, or a fit that cannot be trained, is skipped and logged."""
    r = int(splits_loco["repeat"][0])
    parts = []
    for f in _folds(splits_loco, r):
        train, test = fold_athletes(splits_loco, r, f)
        if len(test) < min_heldout or len(train) < 3 * min_heldout:
            log.warning("[advanced_movement:ATLAS] leave-cohort-out fold %s skipped: %d held-out and %d training athletes", f,
                        len(test), len(train))
            continue
        for spec, tcfg in plan:
            cs = curves[spec.representation]
            try:
                t = tcn.train_fold(cs, splits_loco, r, f, spec, tcfg, train_seed, device)
                e, _, _ = tcn.tcn_fold_outputs(t, cs, splits_loco, None, replace(ecfg, truncations=()), eval_seed, device)
                parts.append(e)
            except tcn.TcnError as err:
                log.warning("[advanced_movement:ATLAS] leave-cohort-out %s fold %s skipped: %s", spec.method, f, err)
        if fpca_spec is not None:
            cs = curves[fpca_spec.representation]
            try:
                m = fp.fit_fpca(cs, train, fcfg, f"{fpca_spec.representation}:loco{f}")
                e, _ = rv.fpca_fold_outputs(fpca_spec, cs, m, r, int(f), train, test, m.support, replace(ecfg, truncations=()), eval_seed)
                parts.append(e)
            except fp.FpcaError as err:
                log.warning("[advanced_movement:ATLAS] leave-cohort-out %s fold %s skipped: %s", fpca_spec.method, f, err)
    return pl.concat(parts) if parts else rv.EMBEDDINGS.empty()
