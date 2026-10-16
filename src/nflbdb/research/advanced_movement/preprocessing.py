"""Fold-safe channel normalisation.

A scaler is fitted on the sequences of the TRAINING athletes of a fold and then applied to everyone. Statistics are
computed over real, defined frames only (padding and undefined values never enter a mean), with every frame of the
training athletes weighted equally. The fitted object records which athletes it saw, and refuses to be reported as
inductive for an athlete it was fitted on.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from .data import assert_disjoint
from .sequences import SequenceError, SequenceSet


@dataclass
class ChannelScaler:
    channels: tuple[str, ...]
    mean: np.ndarray
    std: np.ndarray
    n_frames: np.ndarray
    fit_athletes: tuple[int, ...]

    @classmethod
    def fit(cls, ss: SequenceSet, train_athletes: Sequence[int]) -> "ChannelScaler":
        ids = np.asarray(sorted({int(i) for i in train_athletes}), dtype=np.int64)
        rows = np.isin(ss.nfl_ids, ids)
        if not rows.any():
            raise SequenceError("no sequence of the training athletes: cannot fit the channel scaler")
        X, valid = ss.X[rows].astype(np.float64), ss.valid[rows]
        k = valid.sum((0, 1)).astype(float)
        if (k < 2).any():
            raise SequenceError(f"channel(s) {[c for c, n in zip(ss.channels, k) if n < 2]} have no defined training value")
        mean = (X * valid).sum((0, 1)) / k
        var = (((X - mean) ** 2) * valid).sum((0, 1)) / (k - 1)
        std = np.sqrt(var)
        return cls(ss.channels, mean, np.where(std > 1e-12, std, 1.0), k, tuple(ids.tolist()))

    def transform(self, ss: SequenceSet) -> SequenceSet:
        """Standardised copy. Padded and undefined cells stay exactly zero (which is the training mean)."""
        if ss.channels != self.channels:
            raise SequenceError(f"channel mismatch: scaler fitted on {self.channels}, sequences have {ss.channels}")
        X = np.where(ss.valid, (ss.X - self.mean) / self.std, 0.0).astype(np.float32)
        return SequenceSet(ss.seq_ids, ss.nfl_ids, ss.channels, X, ss.mask, ss.valid, ss.t, ss.boundary, ss.meta)

    def assert_unseen(self, athletes: Sequence[int], where: str = "") -> None:
        """Raise if any of ``athletes`` contributed to this scaler (use before an inductive evaluation)."""
        assert_disjoint(self.fit_athletes, list(athletes), where or "channel scaler vs evaluation athletes")

    def to_dict(self) -> dict[str, Any]:
        return {"channels": list(self.channels), "mean": self.mean.tolist(), "std": self.std.tolist(),
                "n_frames": self.n_frames.tolist(), "fit_athletes": list(self.fit_athletes)}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ChannelScaler":
        return cls(tuple(d["channels"]), np.asarray(d["mean"], float), np.asarray(d["std"], float),
                   np.asarray(d["n_frames"], float), tuple(int(i) for i in d["fit_athletes"]))
