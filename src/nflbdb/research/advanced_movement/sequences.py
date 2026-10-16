"""Sequence dataset: per-frame channels of the eligible primary-drill attempts, and their padded array form.

Source      ``gold/phase_03/combine_frame_kinematics`` (segment-local Savitzky–Golay kinematics, already validated
            in Phase 03/04). Nothing is re-smoothed here, and no operation ever spans two attempts: every derived
            quantity is computed ``over`` the sequence.
Sequence    the valid frames of the LONGEST segment of an attempt — the same selection as the Phase 03 sprint model
            (``sprint_rescue.attempts.longest_segments``), so features and sequences describe the same frames.
Channels    translation- and rotation-invariant by construction. Positions are expressed relative to the first
            frame and to the straight line from the first to the last frame of the sequence:
                along    progress along that line [yd]            lat_dev   sideways deviation from it [yd]
                dist     smoothed path length since frame 0 [yd]    head_sin, head_cos   direction of motion relative
                                                                    to the line (a circular quantity, never averaged
                                                                    as an angle)
            ``a_lat``, ``lat_dev``, ``yaw_rate`` and ``head_sin`` change sign under a mirror image of the run; whether
            the Combine coordinate frame has a fixed handedness is unverified (sprint rescue, limitation), so their
            SIGN is not interpretable across cohorts.
Padding     arrays are zero-padded to the longest sequence. ``mask`` marks real frames; ``valid`` marks real AND
            defined values per channel. A padded cell is never a physical observation: every consumer must use the
            masks, and :meth:`SequenceSet.validate` refuses arrays whose padded cells are not exactly zero.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import polars as pl

from ..contracts import TableContract, cast_keys
from ..sprint_rescue.attempts import longest_segments
from .config import CHANNEL_NAMES, AdvancedConfig
from .data import KEYS

S, I, F, B = pl.String(), pl.Int64(), pl.Float64(), pl.Boolean()
SOURCE_COLUMNS = (*KEYS, "drill", "seg_idx", "pos", "seg_len", "t_s", "x_s", "y_s", "speed", "a_tan", "a_lat", "yaw_rate",
                  "heading", "step_smooth", "pos_resid", "valid_kin", "heading_valid")
SEQ_FRAMES = TableContract(
    "sequence_frames", "Per-frame channels of the eligible primary-drill sequences (longest valid segment).",
    {"seq_id": S, "nfl_id": I, "frame": I, "t_s": F, "pos": I, "seg_len": I, **{c: F for c in CHANNEL_NAMES},
     "boundary": B, "heading_valid": B}, key=("seq_id", "frame"), kind="dataset", area="sequence × frame",
    docs={"frame": "0-based position inside the sequence", "t_s": "ORIGINAL clock: seconds since the trajectory's first "
          "valid frame (Phase 03)", "pos": "original frame position inside its segment", "t_rel": "t_s − t_s of frame 0",
          "boundary": "within the Savitzky–Golay half-window of a segment edge (derivatives less reliable there)"})


class SequenceError(ValueError):
    """The sequence data violate a structural requirement (length, identifiers, padding, channels)."""


# ============================================================================== frames → channels
def build_sequence_frames(frames: pl.LazyFrame | pl.DataFrame, manifest: pl.DataFrame, acfg: AdvancedConfig,
                          half_window: int) -> pl.DataFrame:
    """``frames``: Phase 03 frame kinematics. ``manifest``: :data:`data.MANIFEST`; only ``eligible_primary`` rows are
    kept. ``half_window``: Savitzky–Golay half-window of the persisted kinematics (frames)."""
    lf = frames.lazy()
    missing = [c for c in SOURCE_COLUMNS if c not in lf.collect_schema().names()]
    if missing:
        raise SequenceError(f"frame kinematics lack column(s) {missing}")
    f = cast_keys(lf.filter(pl.col("drill") == acfg.primary_drill).select(SOURCE_COLUMNS).collect())
    f = longest_segments(f, acfg.primary_drill)
    keep = manifest.filter(pl.col("eligible_primary")).select(*KEYS, "seq_id")
    f = f.join(keep, on=KEYS, how="inner").sort("seq_id", "t_s")
    if f.is_empty():
        raise SequenceError("no frame belongs to an eligible primary sequence")
    w = "seq_id"
    f = f.with_columns(pl.int_range(pl.len(), dtype=pl.Int64).over(w).alias("frame"),
                       *[pl.col(c).cast(pl.Float64) for c in ("t_s", "x_s", "y_s", "speed", "a_tan", "a_lat", "yaw_rate",
                                                              "heading", "step_smooth", "pos_resid")])
    f = f.with_columns((pl.col("t_s") - pl.col("t_s").first().over(w)).alias("t_rel"),
                       (pl.col("x_s") - pl.col("x_s").first().over(w)).alias("_dx"),
                       (pl.col("y_s") - pl.col("y_s").first().over(w)).alias("_dy"),
                       pl.when(pl.col("frame") == 0).then(0.0).otherwise(pl.col("step_smooth").fill_null(0.0)).alias("_step"))
    f = f.with_columns(pl.arctan2(pl.col("_dy").last().over(w), pl.col("_dx").last().over(w)).alias("_theta"),
                       pl.col("_step").cum_sum().over(w).alias("dist"))
    rel = pl.col("heading").radians() - pl.col("_theta")
    f = f.with_columns(
        (pl.col("_dx") * pl.col("_theta").cos() + pl.col("_dy") * pl.col("_theta").sin()).alias("along"),
        (pl.col("_dy") * pl.col("_theta").cos() - pl.col("_dx") * pl.col("_theta").sin()).alias("lat_dev"),
        pl.when(pl.col("heading_valid")).then(rel.sin()).otherwise(None).alias("head_sin"),
        pl.when(pl.col("heading_valid")).then(rel.cos()).otherwise(None).alias("head_cos"),
        pl.when(pl.col("heading_valid")).then(pl.col("yaw_rate")).otherwise(None).alias("yaw_rate"),
        ((pl.col("pos") < half_window) | (pl.col("pos") >= pl.col("seg_len") - half_window)).alias("boundary"))
    out = f.select([pl.col(c).cast(t) for c, t in SEQ_FRAMES.columns.items()]).sort("seq_id", "frame")
    short = out.group_by("seq_id").len().filter(pl.col("len") < acfg.min_frames)
    if short.height:
        raise SequenceError(f"{short.height} eligible sequence(s) are shorter than {acfg.min_frames} frames after "
                            f"selecting the longest valid segment, e.g. {short['seq_id'].head(3).to_list()}")
    return out


# ============================================================================== padded arrays
def pad_batch(arrays: Sequence[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Stack (L_i, C) arrays into (n, L_max, C) with zero padding and a boolean (n, L_max) mask of real frames."""
    if not arrays:
        raise SequenceError("cannot pad an empty batch")
    lengths = [a.shape[0] for a in arrays]
    if min(lengths) == 0:
        raise SequenceError("zero-length sequence in batch")
    C = arrays[0].shape[1]
    if any(a.ndim != 2 or a.shape[1] != C for a in arrays):
        raise SequenceError("sequences of one batch must share the channel dimension")
    X = np.zeros((len(arrays), max(lengths), C), dtype=np.float32)
    mask = np.zeros((len(arrays), max(lengths)), dtype=bool)
    for i, a in enumerate(arrays):
        X[i, : a.shape[0]] = a
        mask[i, : a.shape[0]] = True
    return X, mask


@dataclass
class SequenceSet:
    """Padded sequences. ``X`` (n, L, C) float32 · ``mask`` (n, L) real frames · ``valid`` (n, L, C) defined values ·
    ``t`` (n, L) original clock [s] · ``boundary`` (n, L) · ``meta`` one manifest row per sequence, same order."""
    seq_ids: np.ndarray
    nfl_ids: np.ndarray
    channels: tuple[str, ...]
    X: np.ndarray
    mask: np.ndarray
    valid: np.ndarray
    t: np.ndarray
    boundary: np.ndarray
    meta: pl.DataFrame

    @classmethod
    def from_frames(cls, frames: pl.DataFrame, manifest: pl.DataFrame, channels: Sequence[str] = CHANNEL_NAMES) -> "SequenceSet":
        unknown = [c for c in channels if c not in frames.columns]
        if unknown:
            raise SequenceError(f"unknown channel(s) {unknown}; available: {[c for c in CHANNEL_NAMES if c in frames.columns]}")
        f = frames.sort("seq_id", "frame")
        ids = f["seq_id"].to_numpy()
        uniq, start, counts = np.unique(ids, return_index=True, return_counts=True)
        vals = np.column_stack([f[c].cast(pl.Float64).fill_null(float("nan")).to_numpy() for c in channels])
        tt, bb = f["t_s"].to_numpy().astype(float), f["boundary"].fill_null(False).to_numpy()
        n, L, C = uniq.size, int(counts.max()), len(channels)
        X, valid = np.zeros((n, L, C), np.float32), np.zeros((n, L, C), bool)
        mask, t, boundary = np.zeros((n, L), bool), np.zeros((n, L), np.float32), np.zeros((n, L), bool)
        for i, (s, k) in enumerate(zip(start, counts)):
            block = vals[s:s + k]
            ok = np.isfinite(block)
            X[i, :k], valid[i, :k], mask[i, :k] = np.where(ok, block, 0.0), ok, True
            t[i, :k], boundary[i, :k] = tt[s:s + k], bb[s:s + k]
        meta = pl.DataFrame({"seq_id": uniq.astype(str)}).join(manifest, on="seq_id", how="left", maintain_order="left")
        if meta["nfl_id"].null_count():
            raise SequenceError("sequence(s) without a manifest row: rebuild the manifest and the sequence frames together")
        out = cls(uniq.astype(str), meta["nfl_id"].to_numpy().astype(np.int64), tuple(channels), X, mask, valid, t, boundary, meta)
        out.validate()
        return out

    # ------------------------------------------------------------------ structure
    @property
    def n(self) -> int:
        return int(self.seq_ids.size)

    @property
    def lengths(self) -> np.ndarray:
        return self.mask.sum(1)

    def validate(self) -> None:
        n, L, C = self.X.shape
        if (n, L) != self.mask.shape or (n, L, C) != self.valid.shape or C != len(self.channels):
            raise SequenceError(f"shape mismatch: X {self.X.shape}, mask {self.mask.shape}, valid {self.valid.shape}, "
                                f"{len(self.channels)} channels")
        if np.unique(self.seq_ids).size != n:
            raise SequenceError("duplicate sequence identifiers")
        if (self.lengths == 0).any():
            raise SequenceError("zero-length sequence")
        if (self.mask[:, 1:] & ~self.mask[:, :-1]).any():
            raise SequenceError("invalid padding mask: real frames must be contiguous from frame 0")
        if (self.valid & ~self.mask[:, :, None]).any() or np.abs(self.X[~self.valid]).max(initial=0.0) != 0.0:
            raise SequenceError("padded or undefined cells must be exactly zero and flagged invalid")
        if not np.isfinite(self.X).all():
            raise SequenceError("non-finite value inside the array (undefined values must be masked, not stored)")

    def channel(self, name: str) -> int:
        if name not in self.channels:
            raise SequenceError(f"channel {name!r} is not in this set {self.channels}")
        return self.channels.index(name)

    def index_of(self, seq_id: str) -> int:
        hit = np.flatnonzero(self.seq_ids == seq_id)
        if not hit.size:
            raise KeyError(f"sequence {seq_id!r} not in this set")
        return int(hit[0])

    def series(self, i: int, name: str) -> np.ndarray:
        """Values of one channel of sequence ``i`` over its real frames (NaN where undefined)."""
        c, k = self.channel(name), int(self.lengths[i])
        return np.where(self.valid[i, :k, c], self.X[i, :k, c], np.nan).astype(float)

    def subset(self, idx: np.ndarray) -> "SequenceSet":
        idx = np.asarray(idx)
        idx = np.flatnonzero(idx) if idx.dtype == bool else idx
        L = int(self.mask[idx].sum(1).max()) if idx.size else 0
        return SequenceSet(self.seq_ids[idx], self.nfl_ids[idx], self.channels, self.X[idx, :L], self.mask[idx, :L],
                           self.valid[idx, :L], self.t[idx, :L], self.boundary[idx, :L], self.meta[idx.tolist()])

    def for_athletes(self, athlete_ids: Sequence[int]) -> "SequenceSet":
        return self.subset(np.isin(self.nfl_ids, np.asarray(list(athlete_ids), dtype=np.int64)))

    def select_channels(self, names: Sequence[str]) -> "SequenceSet":
        j = [self.channel(c) for c in names]
        return SequenceSet(self.seq_ids, self.nfl_ids, tuple(names), self.X[:, :, j], self.mask, self.valid[:, :, j],
                           self.t, self.boundary, self.meta)

    def channel_missingness(self) -> pl.DataFrame:
        """Share of real frames on which each channel is undefined."""
        real = int(self.mask.sum())
        return pl.DataFrame([{"channel": c, "n_frames": real,
                              "share_undefined": float(1.0 - self.valid[:, :, j].sum() / real) if real else None}
                             for j, c in enumerate(self.channels)])
