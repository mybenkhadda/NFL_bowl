"""Scouting-oriented DESCRIPTIVE views (notebook section 09): forty-yard speed profiles. Nothing here is a recommendation.

    by_outcome_quartile   within one position, mean forty speed curve (25–75 % band) of athletes grouped by the quartile of
                          an NFL target — does the shape of the forty differ between players who later move differently?
    stopwatch_twins       pairs of same-position athletes with |Δforty| ≤ ``twin_s`` whose NFL target differs most — the
                          official time is identical, is the movement profile?

Curves are aligned at the first valid frame (nominal 0.1 s step) from the memory-mapped bundle (bounded slices,
``m15.data.physical_rows``). Athletes are shown by salted pseudonym (``stats.pseudonym``), never by name or id.
"""

from __future__ import annotations

from typing import Any

import numpy as np

NAN = float("nan")
DT = 0.1


def _aligned_speed(bundle: Any, rows: np.ndarray, max_frames: int = 64) -> np.ndarray:
    from ..m15.data import physical_rows

    out = np.full((rows.size, max_frames), NAN)
    if not rows.size:
        return out
    p = physical_rows(bundle, rows)
    sp, ok = p["speed"], p["valid"]
    for k in range(rows.size):
        v = np.flatnonzero(ok[k])
        if v.size:
            seg = sp[k, v[0]:]
            out[k, : seg.size] = seg[:max_frames]
    return out


def by_outcome_quartile(data: Any, feat: Any, targets: Any, target_col: str, position: str, kind: str = "forty", max_rows: int = 600, seed: int = 0) -> Any:
    import polars as pl

    t = targets.select(pl.col("nfl_id").cast(pl.Int64), "position_group", pl.col(target_col).cast(pl.Float64).alias("y")).filter(
        (pl.col("position_group") == position) & pl.col("y").is_finite())
    if t.height < 20:
        return pl.DataFrame()
    q = np.quantile(t["y"].to_numpy(), [0.25, 0.5, 0.75])
    t = t.with_columns(pl.Series("quartile", np.searchsorted(q, t["y"].to_numpy(), side="right") + 1))
    rows = feat.filter((pl.col("domain") == "combine") & (pl.col("kind") == kind)).select("idx", pl.col("nfl_id").cast(pl.Int64)).join(t, on="nfl_id")
    if rows.height > max_rows:
        rows = rows.sample(max_rows, seed=seed)
    S = _aligned_speed(data.bundle, rows["idx"].to_numpy())
    out = []
    for qq in range(1, 5):
        m = rows["quartile"].to_numpy() == qq
        if m.sum() < 5:
            continue
        with np.errstate(all="ignore"):
            mu, lo, hi = np.nanmean(S[m], 0), np.nanquantile(S[m], 0.25, 0), np.nanquantile(S[m], 0.75, 0)
        n_t = np.isfinite(S[m]).sum(0)
        keep = n_t >= 5
        out.append(pl.DataFrame({"label": f"{target_col} Q{qq} ({int(m.sum())} attempts)", "t": np.arange(S.shape[1])[keep] * DT, "speed": mu[keep], "lo": lo[keep],
                                 "hi": hi[keep], "n_attempts": n_t[keep]}))
    return pl.concat(out) if out else pl.DataFrame()


def stopwatch_twins(data: Any, feat: Any, targets: Any, athletes: Any, target_col: str, position: str, twin_s: float = 0.03, n_pairs: int = 3,
                    salt: str = "m16") -> tuple[Any, Any]:
    """(long speed curves of the selected pairs, pair table). Pairs: same position, |Δforty| ≤ twin_s, largest |Δ target|."""
    import polars as pl

    from ..stats import pseudonym

    a = athletes.select(pl.col("nfl_id").cast(pl.Int64), pl.col("forty").cast(pl.Float64)).join(
        targets.select(pl.col("nfl_id").cast(pl.Int64), "position_group", pl.col(target_col).cast(pl.Float64).alias("y")), on="nfl_id").filter(
        (pl.col("position_group") == position) & pl.col("forty").is_finite() & pl.col("y").is_finite())
    ids, f, y = a["nfl_id"].to_numpy(), a["forty"].to_numpy(), a["y"].to_numpy()
    pairs = []
    for i in range(ids.size):
        for j in range(i + 1, ids.size):
            if abs(f[i] - f[j]) <= twin_s:
                pairs.append((abs(y[i] - y[j]), i, j))
    pairs.sort(reverse=True)
    sel, used = [], set()
    for d, i, j in pairs:
        if i in used or j in used:
            continue
        sel.append((i, j, d))
        used |= {i, j}
        if len(sel) >= n_pairs:
            break
    curves, table = [], []
    fr = feat.filter((pl.col("domain") == "combine") & (pl.col("kind") == "forty")).select("idx", pl.col("nfl_id").cast(pl.Int64))
    for k, (i, j, d) in enumerate(sel):
        for who in (i, j):
            r = fr.filter(pl.col("nfl_id") == int(ids[who]))["idx"].to_numpy()
            if not r.size:
                continue
            S = _aligned_speed(data.bundle, r)
            with np.errstate(all="ignore"):
                mu = np.nanmean(S, 0)
            keep = np.isfinite(mu)
            lab = f"pair {k + 1} · {pseudonym(ids[who], salt)} (forty {f[who]:.2f} s, {target_col} {y[who]:.2f})"
            curves.append(pl.DataFrame({"label": lab, "t": np.arange(S.shape[1])[keep] * DT, "speed": mu[keep]}))
        table.append({"pair": k + 1, "a": pseudonym(ids[i], salt), "b": pseudonym(ids[j], salt), "forty_a": float(f[i]), "forty_b": float(f[j]),
                      "target_a": float(y[i]), "target_b": float(y[j]), "abs_target_difference": float(d)})
    return (pl.concat(curves) if curves else pl.DataFrame()), pl.DataFrame(table, infer_schema_length=None)


__all__ = ["by_outcome_quartile", "stopwatch_twins"]
