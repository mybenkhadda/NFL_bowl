"""Athlete folds and train / validation / held-out splits — the 12F notebook rules, unchanged.

* ``fold_map``: the 12.0 fold manifest; athletes absent from it get ``nfl_id % outer_folds`` (12F notebook, Part 3).
* ``split_indices``: delegates to ``se.split_indices`` (held-out athletes of BOTH domains removed; validation = a seeded
  share of the remaining ATHLETES, never random frames).
* ``check_split_integrity``: asserts the grouping the science relies on (no athlete in two roles; held-out = one fold).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .. import _backend

TRAIN, VAL, HELD_OUT = "train", "validation", "held_out"


class SplitLeakError(AssertionError):
    """An athlete appears in more than one split role, or a held-out row is outside the held-out fold."""


def fold_map(meta: Any, folds: Any, outer_folds: int) -> dict[int, int]:
    fmap = dict(zip(*folds.select("nfl_id", "fold").to_dict(as_series=False).values()))
    return {int(i): (fmap[int(i)] if fmap.get(int(i)) is not None else int(i) % outer_folds) for i in meta["nfl_id"].unique().to_list()}


def split_indices(meta: Any, folds_by_athlete: dict[int, int], held_out: int | None, val_share: float, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    tr, va, ho = _backend.se().split_indices(meta, folds_by_athlete, held_out, val_share, seed)
    check_split_integrity(meta, folds_by_athlete, tr, va, ho, held_out)
    return tr, va, ho


def check_split_integrity(meta: Any, folds_by_athlete: dict[int, int], tr: np.ndarray, va: np.ndarray, ho: np.ndarray, held_out: int | None) -> dict[str, Any]:
    ids = meta["nfl_id"].to_numpy()
    n = ids.size
    allidx = np.concatenate([tr, va, ho])
    if allidx.size != n or np.unique(allidx).size != n:
        raise SplitLeakError("train / validation / held-out indices must partition every sequence exactly once")
    a_tr, a_va, a_ho = set(ids[tr].tolist()), set(ids[va].tolist()), set(ids[ho].tolist())
    if a_tr & a_va or a_tr & a_ho or a_va & a_ho:
        raise SplitLeakError(f"athlete(s) in two split roles: {sorted((a_tr & a_va) | (a_tr & a_ho) | (a_va & a_ho))[:5]}")
    if held_out is not None and any(folds_by_athlete.get(int(i), -1) != held_out for i in a_ho):
        raise SplitLeakError("a held-out sequence belongs to an athlete outside the held-out fold")
    if held_out is not None and any(folds_by_athlete.get(int(i), -1) == held_out for i in a_tr | a_va):
        raise SplitLeakError("a training / validation athlete belongs to the held-out fold")
    return {"n_train": int(tr.size), "n_validation": int(va.size), "n_held_out": int(ho.size), "athletes_train": len(a_tr),
            "athletes_validation": len(a_va), "athletes_held_out": len(a_ho)}


def split_roles(n: int, tr: np.ndarray, va: np.ndarray, ho: np.ndarray) -> np.ndarray:
    out = np.full(n, "", dtype=object)
    out[tr], out[va], out[ho] = TRAIN, VAL, HELD_OUT
    return out


__all__ = ["HELD_OUT", "SplitLeakError", "TRAIN", "VAL", "check_split_integrity", "fold_map", "split_indices", "split_roles"]
