"""Part F — objective ablations (mechanism identification) and model complexity.

Ablation names come from the model registry at runtime (``registry.ablations()`` = ``se.ABLATIONS``); nothing is
invented. Every ablation is compared with F4 on the SAME held-out athletes and pools (``compare_inductive`` with
reference F4), and characterised by the same geometry / domain / probe diagnostics.

Physical probes on inductive embeddings are fitted INSIDE the held-out rows of a fold with athlete-grouped inner CV
(those athletes never trained the encoder), so the probe measures what an unseen athlete's embedding encodes.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from ..models import registry as reg

NAN = float("nan")
PROBES = ("max_speed", "mean_a_tan", "max_abs_a_n", "abs_heading_change_rad")


def ablation_names() -> list[str]:
    return list(reg.ablations())


def registry_table() -> Any:
    return reg.list_models()


def inductive_probes(E: np.ndarray, meta: Any, phys: Any, rows: np.ndarray, seed: int, k: int = 5, ridge: float = 1.0, max_rows: int = 6000) -> list[dict[str, Any]]:
    """Athlete-grouped k-fold ridge R² of each physical scalar from the embedding, within ``rows``, per domain."""
    rows = np.sort(np.asarray(rows, np.int64))
    rows = rows[np.isfinite(E[rows]).all(1)]
    rng = np.random.default_rng(seed)
    if rows.size > max_rows:
        rows = np.sort(rng.choice(rows, max_rows, replace=False))
    ph = phys.sort("idx")
    dom = meta["domain"].to_numpy()
    ids = meta["nfl_id"].to_numpy()
    out = []
    for d in ("combine", "nfl"):
        r = rows[dom[rows] == d]
        if r.size < 50:
            continue
        ath = np.unique(ids[r])
        fold_of = dict(zip(rng.permutation(ath).tolist(), (np.arange(ath.size) % k).tolist()))
        f = np.array([fold_of[int(i)] for i in ids[r]])
        Z = E[r].astype(float)
        for p in PROBES:
            y = ph[p].to_numpy().astype(float)[r]
            ok = np.isfinite(y)
            pred = np.full(y.size, np.nan)
            for g in range(k):
                tr, te = (f != g) & ok, (f == g) & ok
                if tr.sum() < 20 or not te.any():
                    continue
                mu, sd = Z[tr].mean(0), np.maximum(Z[tr].std(0), 1e-9)
                Xt = (Z[tr] - mu) / sd
                w = np.linalg.solve(Xt.T @ Xt + ridge * np.eye(Z.shape[1]), Xt.T @ (y[tr] - y[tr].mean()))
                pred[te] = y[tr].mean() + ((Z[te] - mu) / sd) @ w
            m = np.isfinite(pred) & ok
            r2 = float(1 - np.mean((y[m] - pred[m]) ** 2) / np.var(y[m])) if m.sum() > 20 and np.var(y[m]) > 0 else NAN
            out.append({"domain": d, "probe": p, "r2_cv": r2, "n_rows": int(m.sum()), "n_athletes": int(ath.size)})
    return out


def probes_all(exp: Any, models_folds: dict[str, list[int]], seed: int) -> Any:
    import polars as pl

    from .geometry import held_out_rows, inductive_matrix

    meta, phys = exp.bundle.meta, exp.derived.phys
    rows = []
    for n, fl in models_folds.items():
        for f in fl:
            for r in inductive_probes(inductive_matrix(exp, n, f), meta, phys, held_out_rows(exp, f), seed):
                rows.append({"representation": n, "fold": int(f), **r})
    return pl.DataFrame(rows, infer_schema_length=None) if rows else pl.DataFrame()


def training_curves(exp: Any, names: Sequence[str] | None = None) -> Any:
    """All saved per-epoch curves (epoch, val_loss, seconds, train_<loss component>…) of this experiment."""
    import polars as pl

    parts = []
    for p in sorted(exp.store.checkpoints.glob("*/*/curve.parquet")):
        d = pl.read_parquet(p)
        if not d.height:
            continue
        if "model" not in d.columns:
            d = d.with_columns(pl.lit(p.parent.parent.name).alias("model"), pl.lit(p.parent.name).alias("tag"))
        parts.append(d)
    c = pl.concat(parts, how="diagonal_relaxed") if parts else pl.DataFrame()
    if c.height and names:
        c = c.filter(pl.col("model").is_in([reg.resolve(n) for n in names]))
    return c


def curves_long(curves: Any) -> Any:
    """epoch × loss component (train_* and val_loss) in long form for small multiples."""
    import polars as pl

    if curves is None or not curves.height:
        return pl.DataFrame()
    comp = [c for c in curves.columns if c.startswith("train_") or c == "val_loss"]
    return curves.select("model", "tag", "epoch", *comp).unpivot(index=["model", "tag", "epoch"], variable_name="component", value_name="value").with_columns(
        (pl.col("model") + " · " + pl.col("tag")).alias("run"))


def training_health(curves: Any) -> Any:
    """Per run: epochs, best validation loss, finite-loss check, last-epoch time, early stop vs cap (Gate 2 input)."""
    import polars as pl

    if curves is None or not curves.height:
        return pl.DataFrame()
    num = [c for c in curves.columns if c.startswith("train_") or c == "val_loss"]
    return curves.group_by("model", "tag").agg(pl.col("epoch").max().alias("last_epoch"), pl.col("val_loss").min().alias("best_val_loss"),
                                               pl.all_horizontal([pl.col(c).is_finite() for c in num]).all().alias("all_losses_finite"),
                                               pl.col("seconds").sum().alias("train_seconds")).sort("model", "tag")


def complexity_table(exp: Any, names: Sequence[str], curves: Any | None = None) -> Any:
    """Parameters (encoder + heads) of each neural representation and its saved training cost."""
    import polars as pl

    rows = []
    for n in names:
        s = reg.spec(n)
        params = None
        if s.kind == "encoder" or s.name == reg.RANDOM:
            try:
                m = reg.get_model(s.name, exp.cfg.science)
                params = int(sum(p.numel() for p in m.parameters()))
                enc = int(sum(p.numel() for p in m.encoder.parameters()))
            except Exception:                                       # noqa: BLE001 — torch missing → leave empty
                enc = None
        else:
            enc = 0
            params = 0
        secs = None
        if curves is not None and curves.height and "seconds" in curves.columns:
            g = curves.filter(pl.col("model") == s.name)
            secs = float(g["seconds"].sum()) if g.height else None
        rows.append({"representation": s.name, "kind": s.kind, "trainable": s.trainable, "parameters_total": params, "parameters_encoder": enc,
                     "training_seconds_saved": secs, "description": s.description})
    return pl.DataFrame(rows, infer_schema_length=None)


def ablation_summary(comparison: Any, geom_summary: Any, probes: Any, reference: str) -> Any:
    """One row per representation: paired Δ AUC vs F4 (interval), domain AUC, MMD², effective rank, median probe R²,
    and benefit / regression flags versus F4 under declared rules (interval-based for Δ AUC; > 0.02 change for the
    descriptive diagnostics)."""
    import polars as pl

    if comparison is None or not comparison.height:
        return pl.DataFrame()
    c = comparison.filter((pl.col("metric") == "auc")).select("representation", "n_athletes", "mean", "delta", "delta_lo", "delta_hi", "reliability")
    if geom_summary is not None and geom_summary.height:
        g = geom_summary.filter(pl.col("metric").is_in(["domain_auc_grouped", "mmd2", "effective_rank", "position_adjusted_domain_auc"])).pivot(
            on="metric", index="representation", values="mean")
        c = c.join(g, on="representation", how="left")
    if probes is not None and probes.height:
        c = c.join(probes.group_by("representation").agg(pl.col("r2_cv").median().alias("probe_r2_median")), on="representation", how="left")
    ref = c.filter(pl.col("representation") == reference)
    out = []
    for r in c.iter_rows(named=True):
        ben, reg_ = [], []
        if r["representation"] != reference:
            lo, hi = r.get("delta_lo"), r.get("delta_hi")
            if lo is not None and np.isfinite(lo) and lo > 0:
                ben.append("retrieval ↑ (CI)")
            if hi is not None and np.isfinite(hi) and hi < 0:
                reg_.append("retrieval ↓ (CI)")
            if ref.height:
                rr = ref.row(0, named=True)
                for col, better_low, lab in (("domain_auc_grouped", True, "domain separability"), ("mmd2", True, "MMD²"),
                                             ("effective_rank", False, "effective rank"), ("probe_r2_median", False, "physical probes")):
                    a, b = r.get(col), rr.get(col)
                    if a is None or b is None or not (np.isfinite(a) and np.isfinite(b)):
                        continue
                    d = a - b
                    thr = 0.02 if col != "effective_rank" else 1.0
                    if (d < -thr and better_low) or (d > thr and not better_low):
                        ben.append(f"{lab} {'↓' if better_low else '↑'}")
                    elif (d > thr and better_low) or (d < -thr and not better_low):
                        reg_.append(f"{lab} {'↑' if better_low else '↓'}")
        out.append({**r, "benefits_vs_reference": "; ".join(ben) or "—", "regressions_vs_reference": "; ".join(reg_) or "—"})
    return pl.DataFrame(out, infer_schema_length=None)


__all__ = ["PROBES", "ablation_names", "ablation_summary", "complexity_table", "curves_long", "inductive_probes", "probes_all", "registry_table",
           "training_curves", "training_health"]
