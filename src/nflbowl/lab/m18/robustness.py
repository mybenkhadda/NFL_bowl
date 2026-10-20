"""Study E — robustness, negative controls and sanity tests (E18.E1).

Negative controls (fitted with the official engine on the SAME rows / folds; each must NOT beat M1 — a control whose
ΔRMSE CI lies above 0 invalidates the corresponding positive result):
    CTRL_random_embedding   M1 + Gaussian "embeddings" of the same dimension per fold (``se.embedding_block`` path)
    CTRL_shuffled_<rung>    M1 + the best eligible rung's Combine features permuted between athletes WITHIN position
                            (static columns and fold blocks alike): breaks the athlete ↔ movement link, keeps distributions
    CTRL_wrong_drill        M1 + simple features of the drill family NOT matched to the target (Study C mapping)
Sanity: random label — the target permuted within position, B1 and the best rung refitted: |ΔRMSE| must be ≈ 0.
Refit permutation null: ``ae.permutation_blocks`` (within-position permutation of the best rung's extra information,
static and fold-fitted, refitted every time; ``n_perm`` permutations) → p-value.
Sensitivity: draft-year subgroups, complete-case (no missing official test) vs all, high-exposure athletes (≥ median windows /
events of the target) — paired ΔRMSE of the best rung vs M1 inside each subset (descriptive).
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from .ladder import NAMES, emb_factory

NAN = float("nan")


def _shuffle_block(factory: Any, pos: np.ndarray, seed: int) -> Any:
    """A factory whose block rows are permuted within position (same permutation every fold)."""
    rng = np.random.default_rng(seed)
    perm = np.arange(pos.size)
    for p in np.unique(pos):
        ix = np.flatnonzero(pos == p)
        perm[ix] = ix[rng.permutation(ix.size)]

    def f(ids: np.ndarray) -> Any:
        b = factory(ids)
        return lambda tr: b(tr)[perm]
    return f


def control_models(cfg: Any, data: Any, target_id: str, mods: dict[str, Any], best: str | None, fold_of: dict[int, int], blocks: dict[str, Any],
                   analysis_rows: Any, seed: int) -> dict[str, Any]:
    """Control models (official-engine ``ae.Model``) to be fitted next to B1 on the analysis rows of this target."""
    import polars as pl

    from ... import _backend
    from .specificity import DIAGONAL

    ae = _backend.ae()
    m1 = list(ae.M1_COLS)
    out: dict[str, Any] = {NAMES["B1"]: mods[NAMES["B1"]]}
    rng = np.random.default_rng(seed)
    ids = data["nfl_id"].to_numpy()
    rand_bf = {f: pl.DataFrame({"nfl_id": ids, **{f"e{j}": rng.normal(size=ids.size) for j in range(cfg.embedding_dims * 2)}}) for f in set(fold_of.values())}
    out["CTRL_random_embedding"] = ae.Model(m1, factory=emb_factory(fold_of, rand_bf, cfg.embedding_dims))
    if best is not None and best in mods:
        m = mods[best]
        pos = analysis_rows["position_group"].fill_null("?").to_numpy()
        extra = [c for c in m.cols if c not in m1]
        if m.factory is not None:
            out[f"CTRL_shuffled_{best}"] = ae.Model(m1, factory=_shuffle_block(m.factory, pos, seed + 1))
        elif extra:
            perm = np.arange(analysis_rows.height)
            r2 = np.random.default_rng(seed + 1)
            for p in np.unique(pos):
                ix = np.flatnonzero(pos == p)
                perm[ix] = ix[r2.permutation(ix.size)]
            shuffled = {c: analysis_rows[c].to_numpy()[perm] for c in extra}
            out[f"CTRL_shuffled_{best}"] = (extra, shuffled)
    fam_match = {"T1": "forty", "T2": "forty", "T0": "forty", "T3": "agility", "T4": "agility", "T5": "agility"}.get(target_id)
    if fam_match is not None:
        wrong = "agility" if fam_match == "forty" else "forty"
        cols = [c for c in data.columns if c.startswith(f"trk_{wrong}__")]
        if cols and wrong not in DIAGONAL.get(fam_match, ()):
            out["CTRL_wrong_drill"] = ae.Model([*m1, *cols])
    return out


def run_controls(exp: Any, data: Any, target_col: str, ctrl: dict[str, Any], n_boot: int, seed: int) -> Any:
    """Fit controls with the official engine; static shuffled controls are materialised as permuted columns."""
    import polars as pl

    from ... import _backend
    from .. import transfer_fairness as TF

    ae = _backend.ae()
    C = exp.cfg.science.to_m12config()
    d = data.filter(pl.col(target_col).cast(pl.Float64).is_finite())
    mods = {}
    for k, v in ctrl.items():
        if isinstance(v, tuple):
            cols, vals = v
            ren = {c: f"__shuf_{c}" for c in cols}
            d = d.with_columns([pl.Series(ren[c], vals[c][: d.height] if len(vals[c]) == d.height else np.resize(vals[c], d.height)) for c in cols])
            mods[k] = ae.Model([*ae.M1_COLS, *ren.values()])
        else:
            mods[k] = v
    w = ae.oof_ladder(d, target_col, mods, C, C.seed).oof_frame().rename({f"pred_{NAMES['B1']}": "pred_M1"})
    pairs = [(m, "M1") for m in mods if m != NAMES["B1"]]
    p = TF.paired_comparisons(w, pairs, n_boot, seed, "controls")
    if p.height:
        p = p.with_columns(pl.when(pl.col("delta_lo") > 0).then(pl.lit("FAIL (control beats M1)")).otherwise(pl.lit("PASS")).alias("control_status"))
    return p


def random_label(exp: Any, data: Any, target_col: str, mods: dict[str, Any], best: str | None, n_boot: int, seed: int) -> Any:
    import polars as pl

    from ... import _backend
    from .. import transfer_fairness as TF

    ae = _backend.ae()
    C = exp.cfg.science.to_m12config()
    d = data.filter(pl.col(target_col).cast(pl.Float64).is_finite())
    pos = d["position_group"].fill_null("?").to_numpy()
    y = d[target_col].cast(pl.Float64).to_numpy().copy()
    rng = np.random.default_rng(seed)
    for p in np.unique(pos):
        ix = np.flatnonzero(pos == p)
        y[ix] = y[ix][rng.permutation(ix.size)]
    d = d.with_columns(pl.Series("__y_perm", y))
    use = {NAMES["B1"]: mods[NAMES["B1"]]}
    if best and best in mods:
        use[best] = mods[best]
    w = ae.oof_ladder(d, "__y_perm", use, C, C.seed).oof_frame().rename({f"pred_{NAMES['B1']}": "pred_M1"})
    pairs = [(m, "M1") for m in use if m != NAMES["B1"]]
    return TF.paired_comparisons(w, pairs, n_boot, seed, "random_label") if pairs else pl.DataFrame()


def permutation(exp: Any, data: Any, target_col: str, mods: dict[str, Any], best: str | None, n_perm: int, seed: int) -> dict[str, Any]:
    from ... import _backend

    if best is None or best not in mods:
        return {"note": "no eligible rung beyond B1"}
    ae = _backend.ae()
    C = exp.cfg.science.to_m12config()
    r = ae.permutation_blocks(data, target_col, mods[NAMES["B1"]], mods[best], C, seed, n_perm)
    return {"model": best, **r}


def subgroup_sensitivity(wide: Any, analysis: Any, best: str, exposure_col: str | None, n_boot: int, seed: int) -> Any:
    """Paired ΔRMSE (M1 − best) inside draft-year groups, complete-case M1, and high-exposure athletes."""
    import polars as pl

    from .. import transfer_fairness as TF
    from ...data.validation import M1_COLS

    w = wide.join(analysis.select("nfl_id", *[c for c in ("draft_year", *M1_COLS) if c in analysis.columns and c not in wide.columns],
                                  *([exposure_col] if exposure_col and exposure_col in analysis.columns else [])), on="nfl_id", how="left")
    subsets: list[tuple[str, Any]] = []
    if "draft_year" in w.columns:
        subsets += [(f"draft_year={int(v)}", w.filter(pl.col("draft_year") == v)) for v in sorted(w["draft_year"].drop_nulls().unique().to_list())]
    m1 = [c for c in M1_COLS if c in w.columns]
    subsets.append(("complete_case_M1", w.filter(pl.all_horizontal([pl.col(c).is_not_null() for c in m1])) if m1 else w))
    if exposure_col and exposure_col in w.columns:
        med = float(w[exposure_col].median())
        subsets.append((f"{exposure_col} ≥ median", w.filter(pl.col(exposure_col) >= med)))
    rows = []
    for name, s in subsets:
        if s.height < 20:
            rows.append({"subset": name, "n_athletes": s.height, "note": "< 20 athletes"})
            continue
        try:
            p = TF.paired_comparisons(s, [(best, "M1")], n_boot, seed, name)
            rows += [{**r, "subset": name} for r in p.iter_rows(named=True)]
        except TF.ComparisonError as e:
            rows.append({"subset": name, "n_athletes": s.height, "note": str(e)})
    return pl.DataFrame(rows, infer_schema_length=None)


__all__ = ["control_models", "permutation", "random_label", "run_controls", "subgroup_sensitivity"]
