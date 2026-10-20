"""E16.B3 — leakage audit of every M15 representation before it may serve as a Combine-side predictor (M18 B4–B6).

Checks per (representation, fold), read from the M15 run (never modified):
    held_out_disjoint     the fold-f model's training + validation athletes (``train_athletes.json``) contain no athlete
                          whose official fold is f
    roles_match_folds     in the exported table, role == held_out exactly for athletes of official fold f
    bundle_match          the recorded bundle fingerprint equals the current bundle
    config_recorded       ``done.json`` exists with a training-athlete hash equal to the hash of ``train_athletes.json``
A representation is ``combine_only_inductive`` (eligible for M18) only when every check passes on every fold; otherwise
``diagnostic_only``. Alignment heads are fitted inside ``m15.transfer`` on training-role athletes only (by construction).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

from ..status import FAIL, PASS


def _hash(ids: Sequence[int]) -> str:
    return hashlib.sha256(json.dumps(sorted(int(i) for i in ids)).encode()).hexdigest()[:16]


def audit(m15_run_dir: str, reps: Sequence[tuple[str, int]], fold_of: dict[int, int], bundle_fp: str) -> tuple[Any, Any]:
    """(per fold × check table, per representation verdict)."""
    import polars as pl

    from .retrieval import embedding_files

    root = Path(m15_run_dir)
    rows = []
    for arch, d in reps:
        for f, p in embedding_files(m15_run_dir, arch, d).items():
            tag = p.stem.split("__fold")[0]
            ck = root / "checkpoints" / tag / f"fold{f}"
            ta, done = ck / "train_athletes.json", ck / "done.json"
            rec = {"rep": f"{arch}_d{d}", "tag": tag, "fold": f}
            if not ta.is_file() or not done.is_file():
                rows.append({**rec, "check": "config_recorded", "status": FAIL, "detail": f"missing {ta.name if not ta.is_file() else done.name}"})
                continue
            tr = json.loads(ta.read_text(encoding="utf-8"))["train_and_validation_athletes"]
            dn = json.loads(done.read_text(encoding="utf-8"))
            leak = [i for i in tr if fold_of.get(int(i)) == f]
            rows.append({**rec, "check": "held_out_disjoint", "status": PASS if not leak else FAIL, "detail": f"{len(leak)} fold-{f} athlete(s) in training"})
            emb = pl.read_parquet(p, columns=["nfl_id", "role"])
            exp_held = emb["nfl_id"].map_elements(lambda i: fold_of.get(int(i)) == f, return_dtype=pl.Boolean)
            mism = int(((emb["role"] == "held_out") != exp_held).sum())
            rows.append({**rec, "check": "roles_match_folds", "status": PASS if mism == 0 else FAIL, "detail": f"{mism} row(s) with a wrong role"})
            fp = dn.get("bundle_fingerprint")
            rows.append({**rec, "check": "bundle_match", "status": PASS if fp == bundle_fp else FAIL, "detail": f"recorded {fp} vs current {bundle_fp}"})
            rows.append({**rec, "check": "config_recorded", "status": PASS if dn.get("train_athlete_hash") == _hash(tr) else FAIL,
                         "detail": "training-athlete hash consistent" if dn.get("train_athlete_hash") == _hash(tr) else "hash mismatch"})
    tab = pl.DataFrame(rows, infer_schema_length=None)
    if not tab.height:
        return tab, pl.DataFrame(schema={"rep": pl.String, "verdict": pl.String})
    verdict = tab.group_by("rep").agg((pl.col("status") == PASS).all().alias("all_pass"), pl.col("fold").n_unique().alias("n_folds")).with_columns(
        pl.when(pl.col("all_pass")).then(pl.lit("combine_only_inductive")).otherwise(pl.lit("diagnostic_only")).alias("verdict"))
    return tab, verdict


__all__ = ["audit"]
