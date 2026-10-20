"""Study B — pooled (position dummies) vs separate within-position models (E18.B1).

For every position with ≥ ``min_position_n`` athletes in the strict cohort: the same rung models are refitted with the
official engine on that position's athletes only (same outer folds; position dummies become constant). Read-outs per
position × rung: within-position RMSE, Spearman and pairwise concordance for pooled and separate fits; paired ΔRMSE
(pooled − separate) with an athlete bootstrap. A position-specific benefit is claimed only when the separate model is
better with CI > 0 in ≥ 3 positions (gates document); TE is never a primary position.
"""

from __future__ import annotations

from typing import Any

import numpy as np

NAN = float("nan")


def _conc(p: np.ndarray, y: np.ndarray) -> float:
    iu = np.triu_indices(y.size, 1)
    return float(np.mean(np.sign(p[iu[0]] - p[iu[1]]) == np.sign(y[iu[0]] - y[iu[1]]))) if y.size > 2 else NAN


def run_positions(exp: Any, data: Any, target_col: str, mods: dict[str, Any], pooled_wide: Any, strict_ids: set[int], min_n: int, n_boot: int, seed: int) -> Any:
    import polars as pl

    from ... import _backend
    from ..stats import spearman

    ae = _backend.ae()
    C = exp.cfg.science.to_m12config()
    rng = np.random.default_rng(seed)
    rows = []
    d = data.filter(pl.col("nfl_id").is_in(list(strict_ids)))
    for p in sorted(set(d["position_group"].fill_null("?").to_list())):
        sub = d.filter(pl.col("position_group") == p)
        if sub.height < min_n or sub["fold"].n_unique() < 3:
            continue
        try:
            sep = ae.oof_ladder(sub, target_col, mods, C, C.seed).oof_frame()
        except (ValueError, AssertionError):
            continue
        pw = pooled_wide.filter(pl.col("position_group") == p)
        j = sep.join(pw.select("nfl_id", *[c for c in pw.columns if c.startswith("pred_")]), on="nfl_id", suffix="_pooled")
        y = j["y"].to_numpy().astype(float)
        idx = rng.integers(0, y.size, (n_boot, y.size))
        for m in mods:
            ps, pp_col = f"pred_{m}", f"pred_{m}_pooled"          # ``pooled_wide`` = the raw ladder frame (same model names)
            if ps not in j.columns or pp_col not in j.columns:
                continue
            a, b = j[pp_col].to_numpy().astype(float), j[ps].to_numpy().astype(float)
            ok = np.isfinite(a) & np.isfinite(b)
            if ok.sum() < min_n:
                continue
            bs = np.sqrt(((y[idx] - a[idx]) ** 2).mean(1)) - np.sqrt(((y[idx] - b[idx]) ** 2).mean(1))
            rows.append({"position_group": p, "model": m, "n_athletes": int(ok.sum()), "rmse_pooled": float(np.sqrt(np.mean((y - a) ** 2))),
                         "rmse_separate": float(np.sqrt(np.mean((y - b) ** 2))), "delta_pooled_minus_separate": float(np.sqrt(np.mean((y - a) ** 2)) - np.sqrt(np.mean((y - b) ** 2))),
                         "lo": float(np.nanquantile(bs, 0.025)), "hi": float(np.nanquantile(bs, 0.975)), "spearman_pooled": spearman(a, y), "spearman_separate": spearman(b, y),
                         "concordance_pooled": _conc(a, y), "concordance_separate": _conc(b, y)})
    return pl.DataFrame(rows, infer_schema_length=None)


def position_verdict(tab: Any) -> Any:
    import polars as pl

    if tab is None or not tab.height:
        return pl.DataFrame()
    return tab.group_by("model").agg((pl.col("lo") > 0).sum().alias("positions_separate_better"), pl.len().alias("n_positions")).with_columns(
        (pl.col("positions_separate_better") >= 3).alias("position_specific_benefit"))


__all__ = ["position_verdict", "run_positions"]
