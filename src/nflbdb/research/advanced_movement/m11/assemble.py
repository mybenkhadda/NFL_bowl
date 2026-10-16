"""Shared assembly of the Milestone 11 analysis tables from the 11.0 foundation (one definition for every track).

    linked_table     one row per athlete: officials (forty, ten, weight, height, three_cone, short_shuttle), position,
                     draft year, athlete Combine features (wide), NFL targets (wide, supported only), fold
    reliability_map  feature / target → reliability (Combine ICC from attempt repeats, NFL game split-half)
"""

from __future__ import annotations

from typing import Any, Sequence

import polars as pl

from .combine_drills import wide
from .sources import NB
from .speed_state import nfl_wide


def foundation(store: Any, name: str) -> pl.DataFrame:
    return store.read_frame(name, NB["11.0"])


def linked_table(store: Any, extra: Sequence[pl.DataFrame] = (), nfl_required: bool = True, min_episodes: int | None = None) -> pl.DataFrame:
    off = foundation(store, "m11_officials")
    cw = wide(foundation(store, "m11_athlete_features"))
    nw = nfl_wide(foundation(store, "m11_nfl_targets"), supported_only=True, min_episodes=min_episodes)
    folds = foundation(store, "m11_folds").select("nfl_id", "fold", "fold_source")
    d = off.join(cw, on="nfl_id", how="left").join(nw, on="nfl_id", how="inner" if nfl_required else "left")
    for x in extra:
        d = d.join(x, on="nfl_id", how="left")
    return d.join(folds, on="nfl_id", how="inner").sort("nfl_id")


def reliability_map(store: Any) -> dict[str, float]:
    out: dict[str, float] = {}
    rep = foundation(store, "m11_repeatability")
    for r in rep.iter_rows(named=True):
        if r.get("icc") is not None:
            out[r["feature"]] = float(r["icc"])
    rel = foundation(store, "m11_nfl_target_reliability")
    for r in rel.iter_rows(named=True):
        if r.get("reliability") is not None:
            out[r["target_id"]] = float(r["reliability"])
    return out


__all__ = ["foundation", "linked_table", "reliability_map"]
