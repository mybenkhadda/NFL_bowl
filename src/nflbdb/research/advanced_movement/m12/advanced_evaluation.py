"""Evaluation, provenance and checkpoint utilities shared by every Milestone 12 track. NumPy / Polars only.

Reused, not rebuilt: the Milestone 11 train-only design (``m11.evaluation.design``), ridge with an unpenalised position
block (``transfer_models.ridge_solve``), paired athlete bootstrap (``m11.evaluation.paired_compare``), ICC, permutation and
association helpers, immutable checkpoint shards (``m11.checkpoint``).

New here
    * ``Ladder`` / ``oof_ladder``  — out-of-fold predictions where a model may carry a FOLD-FITTED block: a callable
      ``block(train_mask) -> DataFrame`` that fits an unsupervised representation (FPCA, factor model, OT reference …) on
      the training athletes of the fold only and returns scores for every row. Every model of one ladder is scored on the
      SAME athletes (the common cohort of the target), so differences are paired.
    * ``permutation_blocks``       — athlete-label permutation of a block within position. Exact for unsupervised blocks:
      a label-free transform fitted on the training athletes is invariant to the permutation, so permuting its output
      scores equals permuting its inputs.
    * multiplicity (Holm for declared primaries, Benjamini–Hochberg for exploratory families), completion gates,
      memory plans, cohort / fold signatures, evidence rows for the 12G registry.

Δ RMSE = RMSE(reference) − RMSE(model): positive = the model is better (Milestone 11 convention).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import polars as pl

from ..m11 import evaluation as ev
from ..m11.checkpoint import IncompleteRun, ShardCheckpoint
from ..transfer_models import ridge_solve
from .config import M12Config, config_hash

NAN = float("nan")
M0_COLS = ("draft_year",)                                       # + position (always, unpenalised)
M1_COLS = ("draft_year", "height", "weight", "forty", "ten")    # Milestone 11 M1 (context + official core)
M1_PLUS = (*M1_COLS, "three_cone", "short_shuttle")             # sensitivity: official agility times where present


# ============================================================================== grids
def speed_edges(c: M12Config) -> np.ndarray:
    return np.arange(0.0, c.speed_max + c.speed_step / 2, c.speed_step)


def accel_edges(c: M12Config) -> np.ndarray:
    return np.arange(-c.accel_lim, c.accel_lim + c.accel_step / 2, c.accel_step)


def mids(edges: np.ndarray) -> np.ndarray:
    return 0.5 * (edges[:-1] + edges[1:])


def time_grid(c: M12Config) -> np.ndarray:
    return np.round(np.arange(0.0, c.time_max + c.time_step / 2, c.time_step), 6)


# ============================================================================== signatures and checkpoints
def cohort_signature(ids: Sequence[int]) -> str:
    return config_hash(sorted(int(i) for i in ids))


def fold_signature(folds: pl.DataFrame) -> str:
    return config_hash(folds.sort("nfl_id").select("nfl_id", "fold").rows())


def experiment_checkpoint(store: Any, name: str, c: M12Config, cohort: Sequence[int], extra: dict[str, Any] | None = None) -> ShardCheckpoint:
    """Immutable shards keyed by (configuration hash, cohort signature, extra definitions). A shard written under another
    definition is refused (``CheckpointMismatch``) — a legacy checkpoint is never accepted because its rows look right."""
    return ShardCheckpoint(store.shards_dir, name, {"config": c.hash(), "cohort": cohort_signature(cohort), **(extra or {})}, store.controls.run_mode)


def require_complete(ck: ShardCheckpoint, expected: Sequence[Any], what: str) -> None:
    """Completion gate (brief §15.5): final inference tables only when every planned unit is present and verified."""
    st = ck.status(expected)
    if st["n_done"] != st["n_expected"]:
        raise IncompleteRun(f"{what}: {st['n_expected'] - st['n_done']} of {st['n_expected']} planned units are missing (e.g. {st['missing_head']}) — "
                            "partial shards are kept, but no final inference table is produced. Re-run the section to resume.")


# ============================================================================== memory plan (brief §15.2)
def file_mb(path: Path) -> float:
    try:
        return os.stat(path).st_size / 1e6
    except OSError:
        return NAN


def memory_plan(rows: Sequence[dict[str, Any]], budget_gb: float) -> pl.DataFrame:
    """rows: {item, n_rows, n_cols, bytes_per_value, note}. Adds the estimated in-memory size and the configured budget."""
    out = []
    for r in rows:
        mb = float(r.get("n_rows", 0)) * float(r.get("n_cols", 1)) * float(r.get("bytes_per_value", 8)) / 1e6
        out.append({"item": r["item"], "n_rows": int(r.get("n_rows", 0)), "n_cols": int(r.get("n_cols", 1)), "est_mb": round(mb, 1),
                    "on_disk_mb": round(float(r.get("on_disk_mb", NAN)), 1) if r.get("on_disk_mb") is not None else None, "note": r.get("note", "")})
    df = pl.DataFrame(out)
    total = float(df["est_mb"].sum()) if df.height else 0.0
    return df.with_columns(pl.lit(budget_gb * 1000.0).alias("budget_mb"), pl.lit(total <= budget_gb * 1000.0).alias("within_budget"))


# ============================================================================== ladders with fold-fitted blocks
BlockFn = Callable[[np.ndarray], pl.DataFrame]


@dataclass
class Model:
    """``cols``: static columns of the analysis frame; ``block``: optional fold-fitted block (train mask → frame with one row
    per analysis row, same order, numeric columns only); ``factory``: alternative to ``block`` — called with the nfl_id of
    the FINAL analysis rows (after the target / common-cohort filter) and returning the block, so a block can never be
    misaligned with the rows it scores."""
    cols: Sequence[str]
    block: BlockFn | None = None
    note: str = ""
    factory: Callable[[np.ndarray], BlockFn] | None = None

    def resolve(self, ids: np.ndarray) -> BlockFn | None:
        return self.factory(ids) if self.factory is not None else self.block


@dataclass
class LadderResult:
    y: np.ndarray
    ids: np.ndarray
    positions: np.ndarray
    folds: np.ndarray
    preds: dict[str, np.ndarray]
    log: list[dict[str, Any]] = field(default_factory=list)

    def oof_frame(self, **labels: Any) -> pl.DataFrame:
        d = pl.DataFrame({"nfl_id": self.ids, "position_group": self.positions, "fold": self.folds, "y": self.y, **{f"pred_{m}": p for m, p in self.preds.items()}})
        return d.with_columns([pl.lit(str(v)).alias(k) for k, v in labels.items()])


def _xfit(df: pl.DataFrame, cols: Sequence[str], extra: pl.DataFrame | None, train: np.ndarray) -> ev.Design:
    d = df if extra is None or extra.width == 0 else pl.concat([df, extra.rename({k: f"__b_{k}" for k in extra.columns})], how="horizontal")
    use = list(cols) + ([f"__b_{k}" for k in extra.columns] if extra is not None else [])
    return ev.design(d, use, train)


def _fit_predict(df: pl.DataFrame, y: np.ndarray, cols: Sequence[str], extra: pl.DataFrame | None, train: np.ndarray, test: np.ndarray, alpha: float) -> np.ndarray:
    d = _xfit(df, cols, extra, train)
    b0, beta = ridge_solve(d.X[train], y[train], alpha, d.pen)
    return b0 + d.X[test] @ beta


def _alpha(df: pl.DataFrame, y: np.ndarray, cols: Sequence[str], extra: pl.DataFrame | None, train: np.ndarray, c: M12Config, seed: int) -> float:
    """Inner athlete-grouped choice of the ridge penalty among the TRAINING athletes. The fold-fitted block is the one fitted
    on the outer training athletes (it is not refitted per inner fold: the choice of α may be mildly optimistic, the
    held-out athletes are never touched)."""
    idx = np.flatnonzero(train)
    if idx.size < 2 * c.inner_folds:
        return float(max(c.alphas))
    inner = np.random.default_rng(seed).permutation(idx.size) % c.inner_folds
    sse = np.zeros(len(c.alphas))
    for j in range(c.inner_folds):
        tr, va = np.zeros(df.height, bool), np.zeros(df.height, bool)
        tr[idx[inner != j]], va[idx[inner == j]] = True, True
        for k, a in enumerate(c.alphas):
            sse[k] += float(((y[va] - _fit_predict(df, y, cols, extra, tr, va, a)) ** 2).sum())
    best = np.flatnonzero(sse <= sse.min() * (1 + 1e-9))
    return float(c.alphas[int(best.max())])


def oof_ladder(df: pl.DataFrame, target: str, models: dict[str, Model], c: M12Config, seed: int, fold_col: str = "fold",
               require: Sequence[str] = ()) -> LadderResult:
    """Out-of-fold predictions of every model on the rows with a finite ``target`` (and non-null ``require`` columns).
    Folds are athlete folds (``assert_grouped``)."""
    keep = df[target].cast(pl.Float64).is_finite().fill_null(False)
    for k in require:
        keep = keep & df[k].is_not_null()
    d = df.filter(keep)
    y = d[target].cast(pl.Float64).to_numpy()
    folds = d[fold_col].to_numpy()
    ev.assert_grouped(d["nfl_id"].to_numpy(), folds)
    preds = {m: np.full(d.height, NAN) for m in models}
    blocks = {name: m.resolve(d["nfl_id"].to_numpy()) for name, m in models.items()}
    log = []
    for f in sorted(set(folds.tolist())):
        test, train = folds == f, folds != f
        if test.sum() == 0 or train.sum() < 10:
            continue
        for name, m in models.items():
            extra = blocks[name](train) if blocks[name] is not None else None
            if extra is not None and extra.height != d.height:
                raise ValueError(f"block of model {name} returned {extra.height} rows for {d.height} analysis rows")
            a = _alpha(d, y, m.cols, extra, train, c, seed + 31 * int(f))
            preds[name][test] = _fit_predict(d, y, m.cols, extra, train, test, a)
            log.append({"model": name, "fold": int(f), "alpha": a, "n_train": int(train.sum()), "n_test": int(test.sum()),
                        "block_dims": 0 if extra is None else extra.width})
    return LadderResult(y, d["nfl_id"].to_numpy(), d["position_group"].fill_null("?").to_numpy(), folds, preds, log)


def compare(res: LadderResult, comparisons: Sequence[tuple[str, str]], c: M12Config, seed: int, **labels: Any) -> pl.DataFrame:
    rows = []
    for mod, ref in comparisons:
        if mod in res.preds and ref in res.preds:
            r = ev.paired_compare(res.y, res.preds[ref], res.preds[mod], c.n_boot, seed)
            rows.append({**labels, "model": mod, "reference": ref, **r, "verdict": ev.verdict(r)})
    return pl.DataFrame(rows, infer_schema_length=None) if rows else pl.DataFrame()


def compare_by_position(res: LadderResult, mod: str, ref: str, c: M12Config, seed: int, min_n: int | None = None, **labels: Any) -> pl.DataFrame:
    """Same comparison inside each position (subgroup sensitivity; never replaces the pooled primary)."""
    rows = []
    for p in sorted(set(res.positions.tolist())):
        m = res.positions == p
        if m.sum() < (min_n or c.min_athletes // 2):
            continue
        r = ev.paired_compare(res.y[m], res.preds[ref][m], res.preds[mod][m], max(200, c.n_boot // 5), seed)
        rows.append({**labels, "position_group": p, "model": mod, "reference": ref, **r, "verdict": ev.verdict(r)})
    return pl.DataFrame(rows, infer_schema_length=None) if rows else pl.DataFrame()


def permutation_blocks(df: pl.DataFrame, target: str, ref: Model, mod: Model, c: M12Config, seed: int, n_perm: int | None = None,
                       fold_col: str = "fold") -> dict[str, Any]:
    """Null of Δ RMSE (ref − mod) when the model's extra information (static extra columns AND block scores) is permuted
    between athletes of the same position; both the static block and the fold-fitted block are refitted per fold."""
    obs = oof_ladder(df, target, {"ref": ref, "mod": mod}, c, seed, fold_col)
    observed = ev.rmse(obs.y, obs.preds["ref"]) - ev.rmse(obs.y, obs.preds["mod"])
    keep = df[target].cast(pl.Float64).is_finite().fill_null(False)
    d = df.filter(keep)
    base_block = mod.resolve(d["nfl_id"].to_numpy())
    pos = d["position_group"].fill_null("?").to_numpy()
    extra_cols = [k for k in mod.cols if k not in ref.cols]
    rng = np.random.default_rng(seed + 7)
    null = []
    for _ in range(c.n_perm if n_perm is None else n_perm):
        perm = np.arange(d.height)
        for g in np.unique(pos):
            ix = np.flatnonzero(pos == g)
            perm[ix] = rng.permutation(ix)
        dp = d.with_columns([pl.Series(k, d[k].to_numpy()[perm]) for k in extra_cols])
        blk = None
        if base_block is not None:
            blk = (lambda tr, _b=base_block, _p=perm: _b(tr)[_p])
        r = oof_ladder(dp, target, {"mod": Model(mod.cols, blk)}, c, seed, fold_col)
        null.append(ev.rmse(r.y, obs.preds["ref"]) - ev.rmse(r.y, r.preds["mod"]))
    null = np.asarray(null, float)
    return {"observed": float(observed), "n_perm": int(null.size), "null_mean": float(null.mean()) if null.size else NAN,
            "null_q95": float(np.quantile(null, 0.95)) if null.size else NAN,
            "p_value": float((1 + (null >= observed).sum()) / (1 + null.size)) if null.size else NAN}


# ============================================================================== multiplicity
def holm(p: Sequence[float]) -> list[float]:
    return ev.holm([1.0 if (x is None or not np.isfinite(x)) else float(x) for x in p])


def bh(p: Sequence[float]) -> list[float]:
    """Benjamini–Hochberg adjusted p-values (exploratory families)."""
    arr = np.array([1.0 if (x is None or not np.isfinite(x)) else float(x) for x in p])
    n = arr.size
    if n == 0:
        return []
    order = np.argsort(arr)
    adj = np.empty(n)
    run = 1.0
    for r in range(n - 1, -1, -1):
        i = order[r]
        run = min(run, arr[i] * n / (r + 1))
        adj[i] = run
    return adj.tolist()


def add_adjusted(df: pl.DataFrame, p_col: str = "p_boot_le0", how: str = "holm", name: str | None = None) -> pl.DataFrame:
    if df.is_empty() or p_col not in df.columns:
        return df
    p = df[p_col].to_list()
    adj = holm(p) if how == "holm" else bh(p)
    return df.with_columns(pl.Series(name or f"{p_col}_{how}", adj))


# ============================================================================== reliability helpers
def spearman_brown(r: float, k: float) -> float:
    return float(k * r / (1 + (k - 1) * r)) if np.isfinite(r) and r > -1 / max(k - 1, 1e-9) else NAN


def split_reliability(a: np.ndarray, b: np.ndarray, n_boot: int, seed: int) -> dict[str, Any]:
    """Split-half Pearson and Spearman of two independent estimates per athlete, Spearman–Brown corrected, athlete bootstrap."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    n = int(m.sum())
    out = {"n_athletes": n, "r_half": NAN, "rho_half": NAN, "reliability": NAN, "rel_lo": NAN, "rel_hi": NAN}
    if n < 10:
        return out
    a, b = a[m], b[m]
    r = ev.pearson(a, b)
    out |= {"r_half": r, "rho_half": ev.spearman(a, b), "reliability": spearman_brown(r, 2)}
    if n >= ev.MIN_CI_ATHLETES:
        idx = np.random.default_rng(seed).integers(0, n, (n_boot, n))
        bs = np.array([spearman_brown(ev.pearson(a[i], b[i]), 2) for i in idx])
        bs = bs[np.isfinite(bs)]
        if bs.size >= 20:
            out["rel_lo"], out["rel_hi"] = float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))
    return out


def paired_metric_diff(x: np.ndarray, y: np.ndarray, n_boot: int, seed: int, stat: Callable[[np.ndarray], float] = np.mean) -> dict[str, Any]:
    """Athlete-level paired difference stat(x) − stat(y) (e.g. per-athlete losses of two estimators) with a paired bootstrap."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    n = int(m.sum())
    out = {"n_athletes": n, "a": float(stat(x)) if n else NAN, "b": float(stat(y)) if n else NAN, "diff": NAN, "lo": NAN, "hi": NAN}
    if n < 5:
        return out
    out["diff"] = out["a"] - out["b"]
    if n >= ev.MIN_CI_ATHLETES:
        idx = np.random.default_rng(seed).integers(0, n, (n_boot, n))
        d = np.array([stat(x[i]) - stat(y[i]) for i in idx])
        out["lo"], out["hi"] = float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975))
    return out


def reading(lo: float, hi: float, better: str = "positive") -> str:
    if lo is None or hi is None or not (np.isfinite(lo) and np.isfinite(hi)):
        return "insufficient_athletes_for_interval"
    if (lo > 0 and better == "positive") or (hi < 0 and better == "negative"):
        return "supported"
    if (hi < 0 and better == "positive") or (lo > 0 and better == "negative"):
        return "reversed"
    return "inconclusive"


# ============================================================================== evidence registry rows (12G)
EVIDENCE_COLUMNS = ("method_id", "track", "experiment", "physical_interpretation", "input_data", "output", "n_athletes", "positions",
                    "reliability", "reliability_lo", "reliability_hi", "stopwatch_r2", "cross_domain_stat", "cross_domain_lo", "cross_domain_hi",
                    "delta_rmse", "delta_lo", "delta_hi", "opportunity_sensitivity", "compute_cost", "training_regime", "evaluation_target",
                    "evidence_tier", "limitations", "run_mode", "config_hash")


def evidence_row(method_id: str, track: str, **kw: Any) -> dict[str, Any]:
    row = {k: None for k in EVIDENCE_COLUMNS}
    row.update({"method_id": method_id, "track": track, **{k: v for k, v in kw.items() if k in EVIDENCE_COLUMNS}})
    for k in ("reliability", "reliability_lo", "reliability_hi", "stopwatch_r2", "cross_domain_stat", "cross_domain_lo", "cross_domain_hi",
              "delta_rmse", "delta_lo", "delta_hi"):
        v = row[k]
        row[k] = float(v) if v is not None and np.isfinite(float(v)) else None
    row["evidence_tier"] = row["evidence_tier"] or evidence_tier(row)
    return row


def evidence_tier(r: dict[str, Any]) -> str:
    """Declared, ordered rule (never promotes on a point estimate):
    E4 held-out NFL gain (Δ CI > 0) · E3 cross-domain association (CI excludes 0) · E2 reliable (≥ 0.7 with lower bound ≥ 0.5)
    · E1 measurable · E0 not evaluated."""
    if r.get("delta_lo") is not None and r["delta_lo"] > 0:
        return "E4_heldout_physical_gain"
    lo, hi = r.get("cross_domain_lo"), r.get("cross_domain_hi")
    if lo is not None and hi is not None and (lo > 0 or hi < 0):
        return "E3_cross_domain_association"
    if r.get("reliability") is not None and r["reliability"] >= 0.7 and (r.get("reliability_lo") is None or r["reliability_lo"] >= 0.5):
        return "E2_reliable"
    if r.get("n_athletes"):
        return "E1_measurable"
    return "E0_not_evaluated"


def evidence_frame(rows: Sequence[dict[str, Any]]) -> pl.DataFrame:
    schema = {k: (pl.Float64 if k in ("reliability", "reliability_lo", "reliability_hi", "stopwatch_r2", "cross_domain_stat", "cross_domain_lo",
                                      "cross_domain_hi", "delta_rmse", "delta_lo", "delta_hi") else pl.Int64 if k == "n_athletes" else pl.String)
              for k in EVIDENCE_COLUMNS}
    return pl.DataFrame([{k: (None if r.get(k) is None else (str(r[k]) if schema[k] == pl.String else r[k])) for k in EVIDENCE_COLUMNS} for r in rows],
                        schema=schema)


__all__ = ["EVIDENCE_COLUMNS", "LadderResult", "M0_COLS", "M1_COLS", "M1_PLUS", "Model", "accel_edges", "add_adjusted", "bh", "cohort_signature", "compare",
           "compare_by_position", "evidence_frame", "evidence_row", "evidence_tier", "experiment_checkpoint", "file_mb", "fold_signature", "holm", "memory_plan",
           "mids", "oof_ladder", "paired_metric_diff", "permutation_blocks", "reading", "require_complete", "speed_edges", "spearman_brown", "split_reliability",
           "time_grid"]
