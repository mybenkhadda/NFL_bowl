"""Synthetic sequence bundle for the nflbowl tests (no real data, no warehouse).

40 athletes in 3 positions; per athlete 2 Combine forty attempts + 1 three-cone attempt and 4 NFL windows of
accelerating / braking demands; L = 16 frames, 6 channels in the 12F physical scaling. An athlete-specific offset is
planted in the speed channel so retrieval has signal to find.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

N_ATH, L = 40, 16
NFL_KINDS = ("accel_low", "accel_mid", "braking", "redirection")


def make_arrays(seed: int = 0) -> tuple[np.ndarray, np.ndarray, pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    rng = np.random.default_rng(seed)
    ids = np.arange(1000, 1000 + N_ATH)
    pos = {int(i): ("WR", "CB", "DL")[k % 3] for k, i in enumerate(ids)}
    off = {int(i): rng.normal(0, 0.15) for i in ids}
    rows, X, M = [], [], []

    def seq(i: int) -> tuple[np.ndarray, np.ndarray]:
        x = np.zeros((L, 6), np.float32)
        n = int(rng.integers(10, L + 1))
        sp = np.clip(np.linspace(0.05, 0.8, n) + off[i] + rng.normal(0, 0.03, n), 0, None)
        x[:n, 0] = sp
        x[:n, 1] = np.r_[0.0, np.diff(sp)] * 4 + rng.normal(0, 0.02, n)
        x[:n, 2] = rng.normal(0, 0.05, n)
        x[:n, 3] = rng.normal(0, 0.05, n)
        x[:n, 4] = 1.0
        x[:n, 5] = 1.0
        m = np.zeros(L, bool)
        m[:n] = True
        return x, m

    for i in ids:
        for k, kind in enumerate(("forty", "forty", "three_cone")):
            x, m = seq(int(i))
            X.append(x)
            M.append(m)
            rows.append({"seq_key": f"c{i}_{k}", "domain": "combine", "nfl_id": int(i), "kind": kind, "season": None, "n_frames": int(m.sum())})
    for i in ids:
        for k, kind in enumerate(NFL_KINDS):
            x, m = seq(int(i))
            X.append(x)
            M.append(m)
            rows.append({"seq_key": f"n{i}_{k}", "domain": "nfl", "nfl_id": int(i), "kind": kind, "season": 2024, "n_frames": int(m.sum())})
    meta = pl.DataFrame(rows, schema_overrides={"season": pl.Int64, "nfl_id": pl.Int64}).with_row_index("idx").with_columns(
        pl.col("idx").cast(pl.Int64), pl.col("nfl_id").replace_strict(pos, return_dtype=pl.String).alias("position_group"))
    folds = pl.DataFrame({"nfl_id": ids[:-2].astype(np.int64), "fold": (np.arange(N_ATH - 2) % 5).astype(np.int64)})   # 2 athletes fall back to id % 5
    athletes = pl.DataFrame({"nfl_id": ids.astype(np.int64), "position_group": [pos[int(i)] for i in ids], "linked": [True] * N_ATH,
                             "draft_year": [2022 + (k % 3) for k in range(N_ATH)], "height": rng.normal(73, 2, N_ATH), "weight": rng.normal(210, 20, N_ATH),
                             "forty": rng.normal(4.6, 0.1, N_ATH), "ten": rng.normal(1.6, 0.05, N_ATH), "nfl.q90_shared": rng.normal(3.0, 0.4, N_ATH)})
    return np.stack(X), np.stack(M), meta, folds, athletes


@pytest.fixture()
def arrays():
    return make_arrays()


@pytest.fixture()
def bundle_dir(tmp_path, arrays):
    from nflbowl.data.export import write_bundle

    X, M, meta, folds, athletes = arrays
    out = tmp_path / "bundle"
    write_bundle(out, X, M, meta, folds, athletes=athletes, source={"kind": "synthetic_test"})
    return out
