"""E16.B1 / E16.B2 — confirmatory five-fold cross-domain retrieval of the M15 representations (read-only), F0 and random.

Representations are read from the M15 run (``embeddings/inductive/<arch>_d<d>_s<seed>__fold<f>.parquet``); which ones are
eligible comes from the M15 gate (``tables/synthesis/m15_gate``: GO, or GO_DESCRIPTIVE when ``allow_descriptive``).
Scoring = ``m15.transfer`` (pools all / same_position / twins, cosine, per-query rank metrics; Procrustes and InfoNCE
fitted on TRAINING athletes of each fold).

Pre-declared primary (Holm over the family of eligible representations): each eligible representation vs F0, same-position
pool, MRR (per-query reciprocal-rank difference, athlete bootstrap). Secondary: top-1 in the ALL pool minus the per-query
position-only expectation 1/n_pos (does it beat "position recovered, random within position"?).

E16.B2 — context sensitivity: held-out sequences of each fold are matched across domains by CEM on entry speed, valid
length and position (``m15.signals.cem_rows`` → ``domain_matching.cem``); athlete means are recomputed from matched rows
only and the same-position retrieval and the domain AUC (``m15.latent.probes``) are re-scored.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np

from ..status import BLOCKED_MISSING_INPUT, ResearchBlocked

NAN = float("nan")


def eligible_representations(m15_run_dir: str, allow_descriptive: bool = False) -> tuple[list[tuple[str, int]], str]:
    """(arch, d) pairs from the M15 gate table; missing gate → BLOCKED_MISSING_INPUT."""
    import polars as pl

    from ..m15.gates import advancing

    p = Path(m15_run_dir) / "tables" / "synthesis" / "m15_gate.parquet"
    if not p.is_file():
        raise ResearchBlocked(BLOCKED_MISSING_INPUT, "M15 gate table not found (run M15 sections B–E and G15)", missing=[str(p)])
    g = pl.read_parquet(p)
    reps = advancing(g, allow_descriptive)
    return reps, f"{len(reps)} representation(s) eligible ({'GO + GO_DESCRIPTIVE' if allow_descriptive else 'GO only'})"


def embedding_files(m15_run_dir: str, arch: str, d: int, seed: int | None = None) -> dict[int, Path]:
    """{fold: path} for one representation (first available seed when ``seed`` is None)."""
    root = Path(m15_run_dir) / "embeddings" / "inductive"
    files = sorted(root.glob(f"{arch}_d{d}_s*__fold*.parquet"))
    if seed is None and files:
        seed = int(files[0].name.split("_s")[1].split("__")[0])
    return {int(p.stem.split("__fold")[1]): p for p in files if f"_s{seed}__" in p.name}


def forty_map(athletes: Any) -> dict[int, float]:
    import polars as pl

    if athletes is None or "forty" not in athletes.columns:
        return {}
    a = athletes.select(pl.col("nfl_id").cast(pl.Int64), pl.col("forty").cast(pl.Float64)).drop_nulls()
    return dict(zip(a["nfl_id"].to_list(), a["forty"].to_list(), strict=True))


def run_retrieval(m15_run_dir: str, reps: Sequence[tuple[str, int]], index: Any, athletes: Any, e0: np.ndarray | None, folds: Sequence[int], seed: int) -> Any:
    """Per-query scores of every eligible representation (unaligned, procrustes, infonce, random), F0 and random."""
    import polars as pl

    from ..m15 import transfer as T

    forty = forty_map(athletes)
    rows = []
    for arch, d in reps:
        for f, p in embedding_files(m15_run_dir, arch, d).items():
            if f not in folds:
                continue
            emb = pl.read_parquet(p)
            try:
                r, _ = T.retrieval_for_embedding(emb, {"rep": f"{arch}_d{d}", "fold": f}, forty, seed=seed + f)
                rows += r
            except ResearchBlocked:
                continue
    if e0 is not None:
        for f in folds:
            emb = T.sequence_matrix_embedding(e0, index, f)
            try:
                r, _ = T.retrieval_for_embedding(emb, {"rep": "F0", "fold": f}, forty, arms=("unaligned", "procrustes", "random"), seed=seed + f)
                rows += r
            except ResearchBlocked:
                continue
    return pl.DataFrame(rows, infer_schema_length=None)


def primary_tests(scores: Any, n_boot: int, seed: int) -> Any:
    """Each representation (unaligned) vs F0 (unaligned), same-position MRR — Holm over the family; plus the position-only
    test in the ALL pool (top-1 − 1/n_pos per query)."""
    import polars as pl

    from ... import _backend

    ae = _backend.ae()
    rows = []
    reps = [r for r in scores["rep"].unique().to_list() if r != "F0"]
    f0 = scores.filter((pl.col("rep") == "F0") & (pl.col("arm") == "unaligned") & (pl.col("pool") == "same_position")).select("fold", "nfl_id", pl.col("rr").alias("rr_f0"))
    rng = np.random.default_rng(seed)
    for rep in reps:
        s = scores.filter((pl.col("rep") == rep) & (pl.col("arm") == "unaligned"))
        j = s.filter(pl.col("pool") == "same_position").join(f0, on=["fold", "nfl_id"])
        if j.height >= 10:
            d = (j["rr"] - j["rr_f0"]).to_numpy()
            bs = d[rng.integers(0, d.size, (n_boot, d.size))].mean(1)
            rows.append({"rep": rep, "test": "same_position MRR vs F0", "n": int(d.size), "estimate": float(d.mean()), "lo": float(np.quantile(bs, 0.025)),
                         "hi": float(np.quantile(bs, 0.975)), "p_boot_le0": float((bs <= 0).mean()), "primary": True})
        a = s.filter(pl.col("pool") == "all")
        if a.height >= 10:
            npos = a.join(s.filter(pl.col("pool") == "same_position").select("fold", "nfl_id", pl.col("n_candidates").alias("n_same")), on=["fold", "nfl_id"], how="left")
            exp_ = 1.0 / (npos["n_same"].fill_null(0).to_numpy() + 1.0)
            d = npos["top1"].to_numpy() - exp_
            bs = d[rng.integers(0, d.size, (n_boot, d.size))].mean(1)
            rows.append({"rep": rep, "test": "all-pool top-1 minus position-only expectation", "n": int(d.size), "estimate": float(d.mean()),
                         "lo": float(np.quantile(bs, 0.025)), "hi": float(np.quantile(bs, 0.975)), "p_boot_le0": float((bs <= 0).mean()), "primary": False})
    out = pl.DataFrame(rows, infer_schema_length=None)
    if out.height:
        prim = out.filter(pl.col("primary"))
        holm = dict(zip(prim["rep"].to_list(), ae.holm(prim["p_boot_le0"].to_list()))) if prim.height else {}
        out = out.with_columns(pl.struct("rep", "primary").map_elements(lambda r: holm.get(r["rep"]) if r["primary"] else None, return_dtype=pl.Float64).alias("p_holm"))
    return out


def cem_sensitivity(m15_run_dir: str, reps: Sequence[tuple[str, int]], feat: Any, athletes: Any, folds: Sequence[int], seed: int, bins: int = 5) -> Any:
    """Same-position retrieval and domain AUC before / after CEM of held-out sequences (entry speed, length, position)."""
    import polars as pl

    from ..m15 import latent as LT
    from ..m15 import transfer as T
    from ..m15.signals import cem_rows

    forty = forty_map(athletes)
    out = []
    for arch, d in reps:
        for f, p in embedding_files(m15_run_dir, arch, d).items():
            if f not in folds:
                continue
            emb = pl.read_parquet(p)
            held = feat.filter(pl.col("fold") == f)
            nfl_m, comb_m, bal = cem_rows(held.filter(pl.col("domain") == "nfl"), held.filter(pl.col("domain") == "combine"), bins, seed + f)
            keep = set(nfl_m["idx"].to_list()) | set(comb_m["idx"].to_list())
            matched = emb.filter((pl.col("role") != "held_out") | pl.col("idx").is_in(list(keep)))
            for cond, e in (("original", emb), ("cem_matched", matched)):
                lab = {"rep": f"{arch}_d{d}", "fold": f, "condition": cond}
                pr = LT.probes(e, lab)
                try:
                    r, _ = T.retrieval_for_embedding(e, lab, forty, arms=("unaligned",), seed=seed + f)
                    sp = [x for x in r if x["pool"] == "same_position"]
                    mrr = float(np.mean([x["rr"] for x in sp])) if sp else NAN
                except ResearchBlocked:
                    mrr = NAN
                out.append({**lab, "domain_auc": pr.get("domain_auc"), "same_position_mrr": mrr, **({f"cem_{k}": v for k, v in bal.items()} if cond == "cem_matched" else {})})
    return pl.DataFrame(out, infer_schema_length=None)


__all__ = ["cem_sensitivity", "eligible_representations", "embedding_files", "forty_map", "primary_tests", "run_retrieval"]
