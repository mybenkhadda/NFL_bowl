"""11B — speed-state transfer and acceleration envelopes.

Primary hypothesis (declared in ``config.STUDIES['S11B']``): Combine acceleration measured through a specific speed band
relates to NFL acceleration through the SAME band more strongly than to the broad season-average target
(``nfl.broad_accel_sust05``, the Milestone 05 primary). Test: within-position partial Spearman (given official forty and
10-yd split) of the matched pair minus that of the broad pair, athlete-bootstrap CI on the SAME athletes.

    PAIRS                 declared Combine ↔ NFL pairs: matched bands, noncomparable band controls, sprint ↔ envelope parameters
    sprint_checks         our sprint fits vs Phase 03 sprint profiles; model 40-yd time vs official; censoring sensitivity
    direct_transfer       association (raw, partial, disattenuated, within-position) + held-out ladder M1 vs M1 + feature
    matched_vs_broad      the primary comparison
    sensitivity_rows      exposure thresholds, first vs later seasons, positions, envelope estimators, onset censoring
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import polars as pl

from . import evaluation as ev
from .config import M11Config

NAN = float("nan")


@dataclass(frozen=True)
class Pair:
    combine: str
    nfl: str
    kind: str          # matched · noncomparable · physics · broad
    note: str


PAIRS: tuple[Pair, ...] = (
    Pair("forty.band_2_4_mean_accel", "nfl.band_2_4", "matched", "same band, same first-crossing rule"),
    Pair("forty.band_4_6_mean_accel", "nfl.band_4_6", "matched", "same band"),
    Pair("forty.band_6_8_mean_accel", "nfl.band_6_8", "matched", "same band"),
    Pair("forty.band_8_10_mean_accel", "env.env_8_10", "matched", "Combine band vs NFL envelope at 9 yd/s (episodes hold no 8→10 band)"),
    Pair("forty.band_2_4_mean_accel", "nfl.band_6_8", "noncomparable", "NEGATIVE CONTROL: early Combine band vs late NFL band"),
    Pair("forty.band_6_8_mean_accel", "nfl.band_2_4", "noncomparable", "NEGATIVE CONTROL: late Combine band vs early NFL band"),
    Pair("forty.band_4_6_mean_accel", "nfl.broad_accel_sust05", "broad", "matched feature vs broad season-average target"),
    Pair("forty.band_6_8_mean_accel", "nfl.broad_accel_sust05", "broad", "matched feature vs broad season-average target"),
    Pair("forty.band_2_4_mean_accel", "nfl.broad_accel_sust05", "broad", "matched feature vs broad season-average target"),
    Pair("forty.phase_A0", "env.lin_A0", "physics", "sprint intercept A0 = Vmax/τ vs NFL linear envelope A0"),
    Pair("forty.phase_V0", "env.lin_V0", "physics", "sprint V0 vs NFL envelope speed intercept"),
    Pair("forty.sprint_vmax", "env.lin_V0", "physics", "time-form Vmax vs NFL envelope V0"),
    Pair("forty.v_max_obs", "nfl.vmax_p90", "physics", "tracked top speed vs observed game high-speed peak"),
    Pair("forty.early_accel", "nfl.accel_low_mean", "physics", "first second after onset vs acceleration from low speed"),
    Pair("forty.band_4_6_mean_accel", "env.env_4_6", "physics", "Combine band vs envelope at 5 yd/s"),
    Pair("forty.band_6_8_mean_accel", "env.env_6_8", "physics", "Combine band vs envelope at 7 yd/s"),
)
PRIMARY_MATCHED = (("forty.band_4_6_mean_accel", "nfl.band_4_6"), ("forty.band_6_8_mean_accel", "nfl.band_6_8"))


def sprint_checks(attempt_long: pl.DataFrame, p03_sprint: pl.DataFrame | None, officials: pl.DataFrame) -> pl.DataFrame:
    """Plausibility of the sprint fits: agreement with Phase 03 (``model_vmax``, ``model_tau_s``), model 40-yd time vs the
    official forty (expected offset: reaction / start detection), and the share of identifiable onsets."""
    f = attempt_long.filter(pl.col("family") == "forty").pivot(on="feature", index=["seq_id", "nfl_id"], values="value")
    rows = []
    if p03_sprint is not None and p03_sprint.height:
        p = p03_sprint.with_columns((pl.col("event_id").cast(pl.String) + "|" + pl.col("nfl_id").cast(pl.String) + "|" + pl.col("attempt").cast(pl.String)).alias("seq_id"))
        j = f.join(p.select("seq_id", *[k for k in ("model_vmax", "model_tau_s") if k in p.columns]), on="seq_id", how="inner")
        for ours, theirs in (("forty.sprint_vmax", "model_vmax"), ("forty.sprint_tau", "model_tau_s"), ("forty.phase_V0", "model_vmax")):
            if ours in j.columns and theirs in j.columns:
                a, b = j[ours].to_numpy().astype(float), j[theirs].cast(pl.Float64).to_numpy().astype(float)
                ok = np.isfinite(a) & np.isfinite(b)
                rows.append({"check": f"{ours} vs Phase 03 {theirs}", "n": int(ok.sum()), "spearman": ev.spearman(a, b),
                             "median_difference": float(np.median(a[ok] - b[ok])) if ok.any() else NAN})
    a = f.group_by("nfl_id").agg(pl.col("forty.model_t40").mean()).join(officials.select("nfl_id", "forty"), on="nfl_id", how="inner") if "forty.model_t40" in f.columns else None
    if a is not None and a.height:
        x, y = a["forty.model_t40"].to_numpy().astype(float), a["forty"].cast(pl.Float64).to_numpy().astype(float)
        ok = np.isfinite(x) & np.isfinite(y)
        rows.append({"check": "model 40-yd time vs official forty (athlete mean)", "n": int(ok.sum()), "spearman": ev.spearman(x, y),
                     "median_difference": float(np.median(x[ok] - y[ok])) if ok.any() else NAN})
    if "forty.sprint_offset_identifiable" in f.columns:
        s = f["forty.sprint_offset_identifiable"].drop_nulls()
        rows.append({"check": "share of attempts with an identifiable onset (offset fitted)", "n": s.len(), "spearman": NAN,
                     "median_difference": float(s.mean()) if s.len() else NAN})
    return pl.DataFrame(rows, schema={"check": pl.String, "n": pl.Int64, "spearman": pl.Float64, "median_difference": pl.Float64})


def merge_linked(officials: pl.DataFrame, *wides: pl.DataFrame, how_nfl: str = "inner") -> pl.DataFrame:
    """Officials ⨝ Combine wide (inner) ⨝ each further wide table (``how_nfl``): one row per linked athlete."""
    out = officials
    for i, w in enumerate(wides):
        out = out.join(w, on="nfl_id", how="inner" if i == 0 else how_nfl)
    return out


def direct_transfer(data: pl.DataFrame, pairs: Sequence[Pair], reliability: dict[str, float], folds: pl.DataFrame, c: M11Config, seed: int,
                    run_heldout: bool = True) -> tuple[pl.DataFrame, pl.DataFrame]:
    """``data``: one row per linked athlete with officials (forty, ten, …), Combine features and NFL targets (wide).
    Returns (association rows, held-out ladder rows)."""
    assoc, held = [], []
    d = data.join(folds.select("nfl_id", "fold"), on="nfl_id", how="inner")
    for p in pairs:
        if p.combine not in d.columns or p.nfl not in d.columns:
            assoc.append({"combine": p.combine, "nfl": p.nfl, "kind": p.kind, "note": p.note, "n_athletes": 0, "status": "not_available"})
            continue
        x, y = d[p.combine].cast(pl.Float64).to_numpy(), d[p.nfl].cast(pl.Float64).to_numpy()
        Z = np.column_stack([d[k].cast(pl.Float64).fill_null(np.nan).to_numpy() for k in ev.OFFICIAL_CORE if k in d.columns])
        r = ev.association(x, y, Z, c.n_boot, seed, reliability.get(p.combine, NAN), reliability.get(p.nfl, NAN), c.reliability_floor,
                           d["position_group"].to_numpy())
        assoc.append({"combine": p.combine, "nfl": p.nfl, "kind": p.kind, "note": p.note, **r, "reading": ev.association_reading(r), "status": "computed"})
        if run_heldout and p.kind in ("matched", "physics", "broad"):
            sub = d.filter(pl.col(p.combine).is_not_null() & pl.col(p.nfl).is_not_null())
            if sub.height >= c.min_athletes_limited:
                lad = {"M1": [*ev.CONTEXT, *ev.OFFICIAL_CORE], "M2": [*ev.CONTEXT, *ev.OFFICIAL_CORE, p.combine]}
                rows, _ = ev.run_ladder(sub, p.nfl, lad, sub["fold"].to_numpy(), c, seed, comparisons=(("M2", "M1"),),
                                        labels={"combine": p.combine, "kind": p.kind})
                held += rows
    return pl.DataFrame(assoc, infer_schema_length=None), pl.DataFrame(held, infer_schema_length=None)


def matched_vs_broad(data: pl.DataFrame, c: M11Config, seed: int) -> pl.DataFrame:
    """PRIMARY: partial Spearman (given official forty, ten) of each Combine band with its matched NFL band minus that with
    the broad target, on athletes having both targets; athlete bootstrap of the difference."""
    rows = []
    for feat, matched in PRIMARY_MATCHED:
        need = [feat, matched, "nfl.broad_accel_sust05", *[k for k in ev.OFFICIAL_CORE if k in data.columns]]
        if not all(k in data.columns for k in need):
            rows.append({"combine": feat, "matched": matched, "n_athletes": 0, "status": "not_available"})
            continue
        d = data.select("nfl_id", *need).drop_nulls()
        d = d.filter(pl.all_horizontal([pl.col(k).cast(pl.Float64).is_finite() for k in need]))
        n = d.height
        x = d[feat].cast(pl.Float64).to_numpy()
        ym, yb = d[matched].cast(pl.Float64).to_numpy(), d["nfl.broad_accel_sust05"].cast(pl.Float64).to_numpy()
        Z = np.column_stack([d[k].cast(pl.Float64).to_numpy() for k in ev.OFFICIAL_CORE if k in d.columns])
        rm, rb = ev.partial_spearman(x, ym, Z), ev.partial_spearman(x, yb, Z)
        lo = hi = NAN
        if n >= ev.MIN_CI_ATHLETES:
            idx = np.random.default_rng(seed).integers(0, n, size=(min(c.n_boot, 1000), n))
            diffs = np.array([ev.partial_spearman(x[i], ym[i], Z[i]) - ev.partial_spearman(x[i], yb[i], Z[i]) for i in idx])
            diffs = diffs[np.isfinite(diffs)]
            if diffs.size >= 20:
                lo, hi = float(np.quantile(diffs, 0.025)), float(np.quantile(diffs, 0.975))
        verdict = ("matched band stronger (interval > 0)" if np.isfinite(lo) and lo > 0 else "broad target stronger (interval < 0)" if np.isfinite(hi) and hi < 0
                   else "inconclusive" if np.isfinite(lo) else "too few athletes")
        rows.append({"combine": feat, "matched": matched, "n_athletes": n, "partial_matched": rm, "partial_broad": rb, "difference": rm - rb,
                     "diff_lo": lo, "diff_hi": hi, "verdict": verdict, "status": "computed"})
    return pl.DataFrame(rows, infer_schema_length=None)


def sensitivity_rows(base: pl.DataFrame, variants: dict[str, pl.DataFrame], pairs: Sequence[Pair], c: M11Config, seed: int) -> pl.DataFrame:
    """Association of the matched / physics pairs under declared variants (each variant = a full wide table)."""
    rows = []
    for name, d in {"primary": base, **variants}.items():
        for p in pairs:
            if p.kind not in ("matched", "physics") or p.combine not in d.columns or p.nfl not in d.columns:
                continue
            r = ev.association(d[p.combine].cast(pl.Float64).to_numpy(), d[p.nfl].cast(pl.Float64).to_numpy(), None, max(200, c.n_boot // 5), seed)
            rows.append({"variant": name, "combine": p.combine, "nfl": p.nfl, "n_athletes": r["n_athletes"], "spearman": r["spearman"],
                         "spearman_lo": r["spearman_lo"], "spearman_hi": r["spearman_hi"]})
    return pl.DataFrame(rows, infer_schema_length=None)


def nfl_wide(targets_long: pl.DataFrame, supported_only: bool = True, min_episodes: int | None = None) -> pl.DataFrame:
    t = targets_long
    if supported_only:
        t = t.filter(pl.col("supported"))
    if min_episodes is not None:
        t = t.filter(pl.col("n_episodes") >= min_episodes)
    return t.pivot(on="target_id", index="nfl_id", values="value") if t.height else pl.DataFrame({"nfl_id": pl.Series([], dtype=pl.Int64)})


def envelope_wide(env: pl.DataFrame, estimator: str = "lin") -> pl.DataFrame:
    """env.lin_A0 / env.lin_V0 / env.env_<band> columns (``estimator`` picks lin · hub · pow · qr for A0 / V0)."""
    if env.is_empty():
        return pl.DataFrame({"nfl_id": pl.Series([], dtype=pl.Int64)})
    cols = [pl.col(f"{estimator}_A0").alias("env.lin_A0"), pl.col(f"{estimator}_V0").alias("env.lin_V0")]
    cols += [pl.col(k).alias(f"env.{k}") for k in env.columns if k.startswith("env_") and not k.endswith(("_lo", "_hi", "_se"))]
    return env.filter(pl.col("supported")).select("nfl_id", *cols)


__all__ = ["PAIRS", "PRIMARY_MATCHED", "Pair", "direct_transfer", "envelope_wide", "matched_vs_broad", "merge_linked", "nfl_wide", "sensitivity_rows",
           "sprint_checks"]
