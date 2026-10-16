"""Part G1 / G2 — training-seed sensitivity and NFL-data-fraction learning curves (GPU, opt-in).

Seeds: a different science seed changes the science hash, so each seed is its OWN nflbowl experiment
(``<experiment_name>_seed<k>``, same bundle, same preset) — artifacts can never mix seeds. Held-out athletes depend
only on the fold manifest, so every seed's athlete scores share the main run's candidate pools and are compared with
the same paired bootstrap.

Learning curves: ``Experiment.train(nfl_fraction < 1)`` is supported for the TRANSDUCTIVE regime only (12F Part 7a).
Results are labelled transductive diagnostics; they are not evidence of data efficiency for unseen athletes.
"""

from __future__ import annotations

from typing import Any, Sequence

from ..models import registry as reg


def seed_experiment(cfg: Any, run: Any, seed: int) -> Any:
    from ..api import Experiment

    ov = {**cfg.experiment_overrides(), "science.seed": int(seed)}
    exp = Experiment(data_dir=cfg.bundle_dir, output_dir=str(run.root / "checkpoints"), config=cfg.base_config, name=f"{cfg.experiment_name}_seed{seed}",
                     overrides=ov)
    exp.prepare()
    return exp


def seed_scores(cfg: Any, run: Any, model: str, seeds: Sequence[int], folds: Sequence[int], train: bool) -> Any:
    """Athlete retrieval scores of ``model`` for every extra seed (trains the missing (seed, fold) encoders only when
    ``train``), labelled ``<model>@seed<k>``."""
    import polars as pl

    from ..evaluation import paired_inductive as pi
    from .environment import train_missing

    n = reg.resolve(model)
    parts = []
    for s in seeds:
        e = seed_experiment(cfg, run, s)
        if train:
            train_missing(e, [n], folds, resume=True)
        for f in folds:
            if e.manifest.find("training", f"{n}__inductive_f{f}") is None:
                continue
            E = e.embed(n, regime="inductive", folds=[f], reuse=True)[f]
            parts.append(pi.athlete_retrieval_scores(f"{n}@seed{s}", E, e.bundle.meta, e.splits(f)[2], f))
    return pl.concat(parts, how="vertical") if parts else pl.DataFrame()


def seed_comparison(main_scores: Any, extra: Any, model: str, n_boot: int, seed: int) -> Any:
    """Paired Δ AUC of every extra-seed run versus the main run of ``model`` (same athletes, same pools)."""
    import polars as pl

    from ..evaluation import paired_inductive as pi

    n = reg.resolve(model)
    if extra is None or not extra.height:
        return pl.DataFrame()
    base = main_scores.filter(pl.col("representation") == n)
    s = pl.concat([base, extra.select(base.columns)], how="vertical")
    names = [n, *extra["representation"].unique(maintain_order=True).to_list()]
    return pi.paired_bootstrap(s, names, n, metric="auc", n_boot=n_boot, seed=seed)


def learning_curve(exp: Any, model: str, fractions: Sequence[float], train: bool, level: str = "standard") -> Any:
    """Transductive retrieval vs share of NFL windows used in training (labelled diagnostic)."""
    import polars as pl

    n = reg.resolve(model)
    rows = []
    for fr in [*fractions, 1.0]:
        tag = "transductive" if fr >= 1.0 else f"lc{int(round(fr * 100))}"
        if exp.manifest.find("training", f"{n}__{tag}") is None:
            if not train:
                continue
            exp.train(n, regime="transductive", nfl_fraction=fr, resume=True)
        r = exp.evaluate(n, level=level, nfl_fraction=fr, reuse=True)
        rows.append(r.with_columns(pl.lit(fr).alias("nfl_fraction_requested"), pl.lit("transductive diagnostic").alias("label")))
    return pl.concat(rows, how="diagonal_relaxed") if rows else pl.DataFrame()


__all__ = ["learning_curve", "seed_comparison", "seed_experiment", "seed_scores"]
