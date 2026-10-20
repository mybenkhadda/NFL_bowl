"""E16.A9 — cohort and data audit of the M16 laboratory (notebook section 02).

Two definitions of a "linked" athlete are kept side by side and never merged:

    linked_flag      ``athletes.parquet`` column ``linked`` (documented: 401 athletes)
    observed_dual    athletes with ≥ 1 Combine AND ≥ 1 NFL tracking window in the bundle index (documented: 399)

Retrieval needs both domains, so it uses ``observed_dual`` only; prediction (the official ladder) keeps the ``linked``
flag of the frozen M1 cohort. Every disagreement is listed per athlete.

Fold integrity: every athlete sits in exactly one outer fold over all of its windows (athlete-disjoint folds), every fold
is populated, and the fold map of the index equals the experiment's ``folds_by_athlete``. A violation is a
``ValidationFailed`` (FAILED_VALIDATION), never a warning.

``DOCUMENTED`` holds the counts reported in the M15–M18 hand-off. They are compared, not asserted: a different bundle can
legitimately differ, and the difference is shown.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np

from ..status import FAIL, PASS, ValidationFailed

DOCUMENTED = {"n_windows": 77125, "n_combine_windows": 6310, "n_nfl_windows": 70815, "n_combine_athletes": 510, "n_observed_dual": 399,
              "n_linked_flag": 401, "n_folds": 5, "max_frames": 64}


def athlete_domains(index: Any) -> Any:
    """One row per athlete in the sequence index: windows per domain, Combine drill kinds, fold, position."""
    import polars as pl

    return index.group_by("nfl_id").agg(pl.col("position_group").first(), pl.col("fold").n_unique().alias("n_fold_values"), pl.col("fold").first(),
                                        (pl.col("domain") == "combine").sum().alias("n_combine"), (pl.col("domain") == "nfl").sum().alias("n_nfl"),
                                        pl.col("kind").filter(pl.col("domain") == "combine").unique().sort().alias("combine_kinds")).with_columns(
        pl.col("nfl_id").cast(pl.Int64), ((pl.col("n_combine") > 0) & (pl.col("n_nfl") > 0)).alias("observed_dual")).sort("nfl_id")


def dual_domain(index: Any, athletes: Any) -> tuple[Any, dict[str, Any]]:
    """(per-athlete reconciliation table, summary). ``linked_flag`` is null when the athletes table is absent."""
    import polars as pl

    per = athlete_domains(index)
    if athletes is not None and "linked" in athletes.columns:
        flag = athletes.select(pl.col("nfl_id").cast(pl.Int64), pl.col("linked").cast(pl.Boolean).fill_null(False).alias("linked_flag"))
        per = per.join(flag, on="nfl_id", how="full", coalesce=True)
    else:
        per = per.with_columns(pl.lit(None, dtype=pl.Boolean).alias("linked_flag"))
    per = per.with_columns(pl.col("observed_dual").fill_null(False), pl.col("n_combine").fill_null(0), pl.col("n_nfl").fill_null(0))
    per = per.with_columns(pl.when(pl.col("linked_flag").is_null()).then(pl.lit("no flag"))
                           .when(pl.col("linked_flag") & pl.col("observed_dual")).then(pl.lit("agree: linked"))
                           .when(~pl.col("linked_flag") & ~pl.col("observed_dual")).then(pl.lit("agree: not linked"))
                           .when(pl.col("linked_flag")).then(pl.lit("flag only (no tracking in both domains)"))
                           .otherwise(pl.lit("observed only (flag false)")).alias("reconciliation"))
    flag_only = per.filter(pl.col("reconciliation") == "flag only (no tracking in both domains)")
    obs_only = per.filter(pl.col("reconciliation") == "observed only (flag false)")
    summary = {"n_athletes_index": int(per.filter((pl.col("n_combine") + pl.col("n_nfl")) > 0).height),
               "n_combine_athletes": int(per.filter(pl.col("n_combine") > 0).height), "n_nfl_athletes": int(per.filter(pl.col("n_nfl") > 0).height),
               "n_linked_flag": int(per["linked_flag"].fill_null(False).sum()), "n_observed_dual": int(per["observed_dual"].sum()),
               "flag_only_ids": [int(i) for i in flag_only["nfl_id"].to_list()], "observed_only_ids": [int(i) for i in obs_only["nfl_id"].to_list()],
               "rule": "retrieval uses observed_dual; prediction keeps the linked flag of the official M1 cohort"}
    return per, summary


def fold_integrity(index: Any, folds_by_athlete: dict[int, int], outer_folds: int, min_dual_per_fold: int = 20) -> Any:
    """Checks of the athlete-disjoint fold system. Raises ``ValidationFailed`` (after building the table) when an athlete
    spans folds or the index disagrees with the experiment fold map; returns the check table otherwise."""
    import polars as pl

    per = athlete_domains(index)
    spans = per.filter(pl.col("n_fold_values") > 1)
    ids = [int(i) for i in per["nfl_id"].to_list()]
    folds = [int(f) for f in per["fold"].to_list()]
    differ = [i for i, f in zip(ids, folds, strict=True) if i in folds_by_athlete and int(folds_by_athlete[i]) != f]
    absent = [i for i in ids if i not in folds_by_athlete]
    sizes = per.group_by("fold").agg(pl.len().alias("n_athletes"), pl.col("observed_dual").sum().alias("n_dual")).sort("fold")
    populated = set(int(f) for f in sizes["fold"].to_list())
    small = sizes.filter(pl.col("n_dual") < min_dual_per_fold)
    rows = [{"check": "every athlete in exactly one fold (all windows)", "status": PASS if not spans.height else FAIL, "value": spans.height,
             "detail": f"athletes spanning folds: {spans['nfl_id'].head(10).to_list()}" if spans.height else ""},
            {"check": "index folds = experiment folds_by_athlete", "status": PASS if not differ else FAIL, "value": len(differ),
             "detail": f"e.g. {differ[:10]}" if differ else f"{len(absent)} index athlete(s) not in folds_by_athlete (not evaluated by the ladder)"},
            {"check": f"folds 0…{outer_folds - 1} all populated", "status": PASS if populated == set(range(outer_folds)) else FAIL,
             "value": len(populated), "detail": f"populated {sorted(populated)}"},
            {"check": f"≥ {min_dual_per_fold} dual-domain athletes per fold", "status": PASS if not small.height else FAIL, "value": small.height,
             "detail": f"small folds {small.to_dicts()}" if small.height else ""}]
    tab = pl.DataFrame(rows, infer_schema_length=None).with_columns(pl.col("value").cast(pl.String))
    if spans.height or differ:
        raise ValidationFailed("fold system is not athlete-disjoint / disagrees with the experiment fold map", {"spanning": spans["nfl_id"].head(50).to_list(),
                                                                                                                 "differing": differ[:50], "table": tab.to_dicts()})
    return tab


def assert_athlete_disjoint(train_ids: Sequence[int], test_ids: Sequence[int], what: str = "split") -> None:
    """Raise ``ValidationFailed`` when one athlete is both a training and an evaluation unit."""
    both = sorted(set(int(i) for i in train_ids) & set(int(i) for i in test_ids))
    if both:
        raise ValidationFailed(f"{what}: {len(both)} athlete(s) in both training and evaluation sets", {"athletes": both[:50]})


def documented_checks(index: Any, summary: dict[str, Any], max_frames: int) -> Any:
    """Observed counts next to the documented hand-off counts (DESCRIPTIVE: a difference is shown, not raised)."""
    import polars as pl

    obs = {"n_windows": index.height, "n_combine_windows": int((index["domain"] == "combine").sum()), "n_nfl_windows": int((index["domain"] == "nfl").sum()),
           "n_combine_athletes": summary["n_combine_athletes"], "n_observed_dual": summary["n_observed_dual"], "n_linked_flag": summary["n_linked_flag"],
           "n_folds": int(index["fold"].n_unique()), "max_frames": int(max_frames)}
    return pl.DataFrame([{"quantity": k, "observed": int(v), "documented": DOCUMENTED.get(k), "matches": DOCUMENTED.get(k) == int(v)} for k, v in obs.items()])


def funnel(dual: Any, athletes: Any, target_col: str, m1_cols: Sequence[str]) -> Any:
    """Cohort funnel: index athletes → Combine tracking → linked flag → dual observed → official target → complete M1 →
    target ∧ M1 ∧ dual (the cohort where prediction and retrieval can both speak)."""
    import polars as pl

    a = athletes.select(pl.col("nfl_id").cast(pl.Int64), *[c for c in (target_col, *m1_cols) if c in athletes.columns]) if athletes is not None else None
    d = dual.join(a, on="nfl_id", how="left") if a is not None else dual
    has_t = pl.col(target_col).is_not_null() if a is not None and target_col in a.columns else pl.lit(False)
    m1 = [c for c in m1_cols if a is not None and c in a.columns]
    has_m1 = pl.all_horizontal([pl.col(c).is_not_null() for c in m1]) if m1 else pl.lit(False)
    steps = [("athletes in the sequence index", (pl.col("n_combine") + pl.col("n_nfl")) > 0), ("with Combine tracking", pl.col("n_combine") > 0),
             ("linked flag", pl.col("linked_flag").fill_null(False)), ("observed in both domains", pl.col("observed_dual")),
             (f"linked flag with {target_col}", pl.col("linked_flag").fill_null(False) & has_t),
             ("… and complete M1", pl.col("linked_flag").fill_null(False) & has_t & has_m1),
             ("… and observed in both domains", pl.col("linked_flag").fill_null(False) & has_t & has_m1 & pl.col("observed_dual"))]
    rows = []
    for k, (name, expr) in enumerate(steps):
        sub = d.filter(expr)
        rows.append({"order": k, "step": name, "n_athletes": sub.height,
                     **{f"n_{p}": int(sub.filter(pl.col("position_group") == p).height) for p in sorted(set(d["position_group"].drop_nulls().to_list()))}})
    return pl.DataFrame(rows, infer_schema_length=None)


def composition(index: Any) -> dict[str, Any]:
    """Domain × kind × position sequence / athlete counts; NFL exposure (windows per athlete) by position; fold × position."""
    import polars as pl

    by_kind = index.group_by("domain", "kind", "position_group").agg(pl.len().alias("n_windows"), pl.col("nfl_id").n_unique().alias("n_athletes")).sort(
        "domain", "kind", "position_group")
    expo = index.filter(pl.col("domain") == "nfl").group_by("nfl_id").agg(pl.col("position_group").first(), pl.len().alias("n_nfl_windows"),
                                                                         pl.col("valid_s").sum().alias("nfl_valid_seconds"))
    expo_summary = expo.group_by("position_group").agg(pl.len().alias("n_athletes"), pl.col("n_nfl_windows").median().alias("median_windows"),
                                                       pl.col("n_nfl_windows").quantile(0.1).alias("q10_windows"),
                                                       pl.col("n_nfl_windows").quantile(0.9).alias("q90_windows"),
                                                       pl.col("nfl_valid_seconds").median().alias("median_valid_seconds")).sort("position_group")
    fold_pos = index.group_by("fold", "position_group").agg(pl.col("nfl_id").n_unique().alias("n_athletes")).sort("fold", "position_group")
    return {"composition_kind": by_kind, "exposure_per_athlete": expo, "exposure_by_position": expo_summary, "fold_position": fold_pos}


def artifact_inventory(cfg: Any) -> Any:
    """Which upstream artifacts exist (read only). Required ones are named with the steps that need them."""
    import polars as pl

    m14, m15 = Path(cfg.m14_run_dir), Path(cfg.m15_run_dir)
    items = [("M14 athlete features", m14 / "features" / "athlete" / "athlete_features.parquet", "B7 / M1+M14_reliable"),
             ("M14 measurement status", m14 / "tables" / "reliability" / "measurement_status.parquet", "B7 / M1+M14_reliable"),
             ("M15 sequence features", m15 / "features" / "signal" / "sequence_time_features.parquet", "targets, simple kinematics (else recomputed)"),
             ("M15 gate", m15 / "tables" / "synthesis" / "m15_gate.parquet", "G15 decision per representation"),
             ("M15 preservation metrics", m15 / "tables" / "preservation" / "metrics.parquet", "E15.C evidence (G15.a/b)"),
             ("M15 robustness", m15 / "tables" / "preservation" / "robustness.parquet", "E15.C2 evidence (G15.f)"),
             ("M15 latent ICC summary", m15 / "tables" / "reliability" / "latent_icc_summary.parquet", "G15.c evidence"),
             ("M15 run configuration", m15 / "config" / "run_config.json", "channel order / scaling provenance")]
    rows = [{"artifact": n, "path": str(p), "exists": p.is_file(), "needed_for": why, "bytes": p.stat().st_size if p.is_file() else None} for n, p, why in items]
    emb = sorted((m15 / "embeddings" / "inductive").glob("*__fold*.parquet")) if (m15 / "embeddings" / "inductive").is_dir() else []
    rows.append({"artifact": "M15 inductive embedding tables", "path": str(m15 / "embeddings" / "inductive"), "exists": bool(emb), "needed_for": "Tracks B and C",
                 "bytes": int(sum(p.stat().st_size for p in emb)) if emb else None})
    for d in cfg.experiment_dirs():
        recs = sorted((Path(d) / "evaluation" / "records").glob("transfer_oof__*.parquet"))
        rows.append({"artifact": f"persisted OOF records ({Path(d).name})", "path": str(Path(d) / "evaluation" / "records"), "exists": bool(recs),
                     "needed_for": "G16-B persisted comparison", "bytes": int(sum(p.stat().st_size for p in recs)) if recs else None})
    for d in cfg.legacy_m15_dirs:
        rows.append({"artifact": "legacy M15 folder", "path": str(d), "exists": Path(d).exists(), "needed_for": "provenance audit (exploratory artifacts)",
                     "bytes": None})
    out = pl.DataFrame(rows, infer_schema_length=None)
    return out.with_columns(pl.Series("n_files", [len(emb) if r["artifact"] == "M15 inductive embedding tables" else None for r in rows], dtype=pl.Int64))


def exposure_quantiles(expo: Any, qs: Sequence[float] = (0.1, 0.25, 0.5, 0.75, 0.9)) -> dict[str, float]:
    v = expo["n_nfl_windows"].to_numpy().astype(float)
    return {f"q{int(100 * q)}": float(np.quantile(v, q)) for q in qs} if v.size else {}


__all__ = ["DOCUMENTED", "artifact_inventory", "assert_athlete_disjoint", "athlete_domains", "composition", "documented_checks", "dual_domain",
           "exposure_quantiles", "fold_integrity", "funnel"]
