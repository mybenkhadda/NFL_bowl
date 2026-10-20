"""Study A — the augmentation ladder B0–B7 on one cohort, one fold system and one target (E18.A1).

    B0  position only                         ``ae.Model([])``
    B1  official Combine M1                   ``ae.Model(M1_COLS)``
    B2  M1 + simple tracking statistics       M16 ``trk_*`` columns, train-coverage factory
    B3  M1 + reliable biomechanics            M14 ``validated`` metrics (M16 ``m14_reliable_table``) ∪ M17-repaired metrics are not
                                              mixed here: B3 uses M14 only; M17 output enters through B7
    B4  M1 + M15 STFT embeddings              ``se.embedding_block`` (≤ ``embedding_dims`` train-fold PCA dims of the fold-f model's
    B5  M1 + M15 LSTM / GRU embeddings        athlete-mean Combine embedding — the official ladder convention)
    B6  M1 + fused embeddings                 M17 fused tables when an arm was adopted, else B4 ⊕ B5 blocks (``combine_factories``)
    B7  M1 + validated M17 signatures         ``sig_*`` columns with G17 status ``validated``

Eligibility comes ONLY from upstream gate tables (``eligible_rungs``): a rung whose evidence is missing or failed is
``blocked`` with the reason and is never fitted. Comparisons: every eligible rung vs B1 (and B1 vs B0), paired fold-
stratified athlete bootstrap; Holm over the PRE-REGISTERED primary family (eligible B2…B7 vs B1 on the primary target);
``m14.transfer.transfer_status`` → validated / exploratory / inconclusive / invalid; practical margin and precise null.

Fold safety: embeddings come from the fold-f model (never trained on fold-f athletes); every block is fitted on training
rows of the outer fold; M17 signatures of fold-f athletes use parameters estimated from the other folds (Combine data of
held-out athletes contributes to the context means used for other folds' training athletes — no outcome is involved;
stated in the model card).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np

from ..status import BLOCKED_MISSING_INPUT, ResearchBlocked

NAN = float("nan")
NAMES = {"B0": "B0_position", "B1": "B1_M1", "B2": "B2_M1+simple", "B3": "B3_M1+M14", "B4": "B4_M1+STFT", "B5": "B5_M1+LSTM", "B6": "B6_M1+fusion",
         "B7": "B7_M1+signatures"}


def _pq(p: Path) -> Any:
    import polars as pl

    return pl.read_parquet(p) if p.is_file() else None


def gate_tables(cfg: Any) -> dict[str, Any]:
    return {"g15": _pq(Path(cfg.m15_run_dir) / "tables" / "synthesis" / "m15_gate.parquet"),
            "g16": _pq(Path(cfg.m16_run_dir) / "tables" / "synthesis" / "m16_gate.parquet"),
            "leak": _pq(Path(cfg.m16_run_dir) / "tables" / "retrieval" / "leakage_verdict.parquet"),
            "g17": _pq(Path(cfg.m17_run_dir) / "tables" / "synthesis" / "m17_gate.parquet"),
            "fusion": _pq(Path(cfg.m17_run_dir) / "tables" / "fusion" / "fusion_adoption.parquet")}


def eligible_rungs(cfg: Any, gates: dict[str, Any], analysis: Any, m14_cols: Sequence[str], sig_cols: Sequence[str]) -> tuple[dict[str, dict[str, Any]], Any]:
    """{rung: {status, reason, ...}} from gate tables only, and the same as a frame."""
    import polars as pl

    from ..m15.gates import advancing

    out: dict[str, dict[str, Any]] = {"B0": {"status": "eligible", "reason": "always"}}
    g16 = gates.get("g16")
    gb = g16.filter(pl.col("criterion").str.starts_with("G16-B")) if g16 is not None else None
    b_ok = gb is not None and gb.height and gb["status"][0] == "PASS"
    out["B1"] = {"status": "eligible" if b_ok else "eligible_unverified", "reason": "G16-B PASS" if b_ok else
                 "G16-B not PASS (M1 reproduction not verified against persisted records): results labelled unverified"}
    trk = [c for c in analysis.columns if c.startswith("trk_")]
    out["B2"] = {"status": "eligible" if trk else "blocked", "reason": f"{len(trk)} simple tracking columns", "columns": trk}
    out["B3"] = {"status": "eligible" if m14_cols else "blocked", "reason": f"{len(m14_cols)} M14 validated metric(s)", "columns": list(m14_cols)}
    reps = advancing(gates["g15"], cfg.allow_descriptive) if gates.get("g15") is not None else []
    leak = gates.get("leak")
    ok_leak = set(leak.filter(pl.col("verdict") == "combine_only_inductive")["rep"].to_list()) if leak is not None and leak.height else None
    reps = [r for r in reps if ok_leak is None or f"{r[0]}_d{r[1]}" in ok_leak]
    stft = [r for r in reps if r[0] == "stft_ae"]
    rnn = [r for r in reps if r[0] in ("lstm_ae", "gru_ae")]
    why15 = "M15 gate table missing" if gates.get("g15") is None else ("no leakage audit (M16 B3) — unverified" if ok_leak is None else "")
    out["B4"] = {"status": "eligible" if stft else "blocked", "reason": f"STFT representation {stft[0]}" if stft else f"no G15-GO STFT representation {why15}".strip(), "rep": stft[0] if stft else None}
    out["B5"] = {"status": "eligible" if rnn else "blocked", "reason": f"recurrent representation {rnn[0]}" if rnn else f"no G15-GO LSTM/GRU representation {why15}".strip(), "rep": rnn[0] if rnn else None}
    fu = gates.get("fusion")
    adopted = fu.filter(pl.col("adopted").fill_null(False))["arm"].to_list() if fu is not None and fu.height else []
    if stft and rnn:
        out["B6"] = {"status": "eligible", "reason": f"fused arm {adopted[0]}" if adopted else "B4 ⊕ B5 blocks (no fusion arm adopted in M17)", "arm": adopted[0] if adopted else None}
    else:
        out["B6"] = {"status": "blocked", "reason": "needs both B4 and B5"}
    out["B7"] = {"status": "eligible" if sig_cols else "blocked", "reason": f"{len(sig_cols)} G17-validated signature(s)" if sig_cols else "no G17-validated signature",
                 "columns": list(sig_cols)}
    frame = pl.DataFrame([{"rung": k, "name": NAMES[k], "status": v["status"], "reason": v["reason"]} for k, v in out.items()])
    return out, frame


def by_fold_tables(m15_run_dir: str, rep: tuple[str, int], kinds: Sequence[str]) -> dict[int, Any]:
    """{fold: athlete-mean Combine embedding (nfl_id, e0…)} of the fold-f model (every athlete, as the official ladder)."""
    import polars as pl

    from ..m15.latent import zcols
    from ..m16.retrieval import embedding_files

    out = {}
    for f, p in embedding_files(m15_run_dir, *rep).items():
        e = pl.read_parquet(p).filter((pl.col("domain") == "combine") & pl.col("kind").is_in(list(kinds)))
        zc = zcols(e)
        out[f] = e.group_by("nfl_id").agg([pl.col(c).mean().alias(f"e{k}") for k, c in enumerate(zc)]).with_columns(pl.col("nfl_id").cast(pl.Int64))
    return out


def fused_by_fold(m17_run_dir: str, arm: str, kinds: Sequence[str]) -> dict[int, Any]:
    import polars as pl

    from ..m15.latent import zcols

    out = {}
    for p in sorted((Path(m17_run_dir) / "features" / "fusion").glob(f"{arm}__fold*.parquet")):
        f = int(p.stem.split("__fold")[1])
        e = pl.read_parquet(p).filter((pl.col("domain") == "combine") & pl.col("kind").is_in(list(kinds)))
        out[f] = e.group_by("nfl_id").agg([pl.col(c).mean().alias(f"e{k}") for k, c in enumerate(zcols(e))]).with_columns(pl.col("nfl_id").cast(pl.Int64))
    return out


def emb_factory(fold_of: dict[int, int], by_fold: dict[int, Any], k: int) -> Any:
    from ... import _backend

    se = _backend.se()
    return lambda ids, _b=by_fold: se.embedding_block(ids, fold_of, _b, k)


def build_models(cfg: Any, rungs: dict[str, dict[str, Any]], data: Any, fold_of: dict[int, int], log: list) -> tuple[dict[str, Any], dict[str, Any]]:
    """(official-engine models keyed by rung name, block tables used — for controls)."""
    from ... import _backend
    from ..m14.transfer import combine_factories, coverage_factory
    from ..m16.baselines import assert_no_nfl_predictors

    ae = _backend.ae()
    m1 = list(ae.M1_COLS)
    mods: dict[str, Any] = {NAMES["B0"]: ae.Model([]), NAMES["B1"]: ae.Model(m1)}
    blocks: dict[str, Any] = {}
    for r in ("B2", "B3"):
        if rungs[r]["status"] == "eligible":
            cols = rungs[r]["columns"]
            assert_no_nfl_predictors(cols)
            mods[NAMES[r]] = ae.Model(m1, factory=coverage_factory(data, cols, 0.5, log, NAMES[r]))
    for r in ("B4", "B5"):
        if rungs[r]["status"] == "eligible":
            bf = by_fold_tables(cfg.m15_run_dir, rungs[r]["rep"], cfg.embedding_kinds)
            blocks[r] = bf
            mods[NAMES[r]] = ae.Model(m1, factory=emb_factory(fold_of, bf, cfg.embedding_dims))
    if rungs["B6"]["status"] == "eligible":
        if rungs["B6"].get("arm"):
            bf = fused_by_fold(cfg.m17_run_dir, rungs["B6"]["arm"], cfg.embedding_kinds)
            blocks["B6"] = bf
            mods[NAMES["B6"]] = ae.Model(m1, factory=emb_factory(fold_of, bf, cfg.embedding_dims))
        else:
            mods[NAMES["B6"]] = ae.Model(m1, factory=combine_factories(emb_factory(fold_of, blocks["B4"], cfg.embedding_dims),
                                                                       emb_factory(fold_of, blocks["B5"], cfg.embedding_dims)))
    if rungs["B7"]["status"] == "eligible":
        cols = [c for c in rungs["B7"]["columns"] if c in data.columns]
        if cols:
            mods[NAMES["B7"]] = ae.Model([*m1, *cols])
    return mods, blocks


def run_ladder(exp: Any, data: Any, target_col: str, mods: dict[str, Any]) -> Any:
    import polars as pl

    from ... import _backend

    ae = _backend.ae()
    C = exp.cfg.science.to_m12config()
    res = ae.oof_ladder(data, target_col, mods, C, C.seed)
    w = res.oof_frame()
    y, f = w["y"].to_numpy(), w["fold"].to_numpy()
    mean = np.full(w.height, NAN)
    for k in np.unique(f):
        mean[f == k] = float(y[f != k].mean())
    return w.with_columns(pl.Series("pred_MEAN", mean)), pl.DataFrame(res.log, infer_schema_length=None)


def evaluate(wide: Any, fold_of: dict[int, int], target_id: str, primary: bool, cfg: Any) -> dict[str, Any]:
    import polars as pl

    from .. import transfer_fairness as TF
    from ..m14.transfer import fold_stability, transfer_status
    from ..m16.baselines import precise_null

    models = [c.removeprefix("pred_") for c in wide.columns if c.startswith("pred_")]
    long = TF.long_predictions(wide, f"m18_{target_id}", models)
    audit = TF.validate_alignment(long, fold_of)
    fin = long.filter(pl.col("pred").is_finite()).group_by("nfl_id").agg(pl.col("model").n_unique().alias("k"))
    strict = set(int(i) for i in fin.filter(pl.col("k") == len(models))["nfl_id"].to_list())
    w = TF.wide(long, strict, models)
    w1 = w.rename({f"pred_{NAMES['B1']}": "pred_M1"})
    ms = [m if m != NAMES["B1"] else "M1" for m in models]
    pairs = [(m, "M1") for m in ms if m not in ("M1", "MEAN", NAMES["B0"])] + [("M1", NAMES["B0"])]
    fam = [(m, "M1") for m in ms if m.startswith(("B2", "B3", "B4", "B5", "B6", "B7"))] if primary else []
    paired = precise_null(TF.paired_comparisons(w1, pairs, cfg.n_boot, cfg.seed, "strict_common", primary_family=fam), cfg.margin_pct / 100)
    if paired.height:
        paired = paired.with_columns((pl.col("delta_pct") >= cfg.margin_pct).alias("exceeds_practical_margin"))
    folds = TF.fold_table(w1, ms, "M1")
    status = transfer_status(paired, fold_stability(folds))
    tag = pl.lit(target_id).alias("target_id")
    return {"wide": w1, "long": long.with_columns(tag), "audit": audit, "strict_ids": strict, "models": ms, "primary_family": fam,
            "metrics": TF.cohort_metrics(w1, ms, "strict_common", position_model=NAMES["B0"]).with_columns(tag),
            "paired": paired.with_columns(tag) if paired.height else paired, "folds": folds.with_columns(tag),
            "status": status.with_columns(tag) if status.height else status, "intervals": TF.rmse_intervals(w1, ms, cfg.n_boot, cfg.seed, "strict_common").with_columns(tag)}


def analysis_with_signatures(cfg: Any) -> Any:
    """The M16 analysis frame (read only) joined with the M17 signature table when present."""
    import polars as pl

    a = _pq(Path(cfg.m16_run_dir) / "datasets" / "cached" / "analysis_frame.parquet")
    if a is None:
        raise ResearchBlocked(BLOCKED_MISSING_INPUT, "M16 analysis frame not found (run M16 E16.A4)", missing=[str(Path(cfg.m16_run_dir) / "datasets/cached/analysis_frame.parquet")])
    sig = _pq(Path(cfg.m17_run_dir) / "features" / "events_v2" / "athlete_signatures.parquet")
    if sig is not None:
        a = a.join(sig.select("nfl_id", *[c for c in sig.columns if c.startswith("sig_")]).with_columns(pl.col("nfl_id").cast(pl.Int64)), on="nfl_id", how="left")
    return a


__all__ = ["NAMES", "analysis_with_signatures", "build_models", "by_fold_tables", "eligible_rungs", "emb_factory", "evaluate", "fused_by_fold", "gate_tables",
           "run_ladder"]
