"""NFL physical targets of Milestone 11 (11A · A2, 11B, 11C · C4) — no Combine value is read here.

From the persisted 09A episodes (no rescan)
    ``TARGETS``              declared athlete-level targets: matched speed bands, acceleration from low speed, high-speed
                             transition, observed top speed, braking, re-acceleration, redirection speed retention,
                             Brake-to-Burst, pair Transition Cost Index; the broad Milestone 05 primary is added from 05A
    ``brake_reaccel_pairs``  a braking episode and the re-acceleration that follows it in the same player-play (≤ 1 s gap);
                             turn angle only when a redirection episode with a valid heading overlaps (else unobserved)
    ``athlete_targets``      median (or declared statistic) per athlete with episode, game and season counts
    ``target_reliability``   game-parity split-half, Spearman–Brown

From ONE resumable scan of the tracking (``EnvelopeBuilder``; Milestone 05 game batches, segmentation and kinematics)
    athlete × game × speed bin × acceleration bin frame counts of eligible post-snap, non-boundary frames — the basis of
    the observed acceleration–speed envelopes (``envelope_table``) with game-resampling uncertainty

For 11F (``NflWindowExtractor``)
    fixed-length frame windows (speed, a_tan, a_n, yaw rate, Δt, validity) around a deterministic, capped sample of episodes,
    written as checkpoint shards per season × batch.

Observed game movement follows role and opportunity: every target is what the athlete DID, never a capacity.
"""

from __future__ import annotations

import gc
import time
from dataclasses import dataclass
from typing import Any, Iterator, Sequence

import numpy as np
import polars as pl

from ....data.utils import get_logger
from ....features.config import SmoothingSpec
from ....features.kinematics import compute_kinematics
from ....features.play_windows import PLAY_KEYS, PP_KEYS
from ....features.trajectory import NFL, prepare_frames
from ..nfl_outcomes import NflOutcomeBuilder, OutcomeConfig
from . import physics as ph
from .checkpoint import ShardCheckpoint
from .config import M11Config, Progress
from .evaluation import split_half

log = get_logger(__name__)
NAN = float("nan")
ACCEL_DEMANDS = ("accel_low", "accel_mid", "transition_high", "reaccel_after_brake")


@dataclass(frozen=True)
class TargetSpec:
    target_id: str
    label: str
    unit: str
    source: str              # "episodes" · "pairs" · "plays"
    demands: tuple[str, ...]
    expr: str                # column or derived expression name (``_value``)
    stat: str                # median · p90
    where: str               # declared physical condition (documentation; applied in ``_values``)
    combine_match: str       # the Combine quantity measured under the same rule ("" = none)
    higher_is: str           # physical direction of a larger value


TARGETS: tuple[TargetSpec, ...] = (
    TargetSpec("nfl.band_2_4", "acceleration through 2 → 4 yd/s", "yd/s²", "episodes", ACCEL_DEMANDS, "acc_2_4", "median",
               "accelerating episodes whose trace crosses 2 and 4 yd/s (first-crossing rule)", "forty.band_2_4_mean_accel", "faster speed build-up"),
    TargetSpec("nfl.band_4_6", "acceleration through 4 → 6 yd/s", "yd/s²", "episodes", ACCEL_DEMANDS, "acc_4_6", "median",
               "same rule, 4 → 6 yd/s", "forty.band_4_6_mean_accel", "faster speed build-up"),
    TargetSpec("nfl.band_6_8", "acceleration through 6 → 8 yd/s", "yd/s²", "episodes", ACCEL_DEMANDS, "acc_6_8", "median",
               "same rule, 6 → 8 yd/s", "forty.band_6_8_mean_accel", "faster speed build-up"),
    TargetSpec("nfl.accel_low_mean", "mean acceleration from low speed", "yd/s²", "episodes", ("accel_low",), "mean_accel", "median",
               "accel_low episodes (start < 2 yd/s)", "forty.early_accel", "harder acceleration from a near-standstill"),
    TargetSpec("nfl.accel_low_peak05", "peak 0.5-s speed gain from low speed", "yd/s²", "episodes", ("accel_low",), "peak_rate_05", "median",
               "accel_low episodes", "forty.early_accel", "harder acceleration"),
    TargetSpec("nfl.transition_high_mean", "acceleration into high-speed running", "yd/s²", "episodes", ("transition_high",), "mean_accel", "median",
               "transition_high episodes (start ≥ 5 yd/s, reach 7.5 yd/s)", "forty.band_6_8_mean_accel", "harder late acceleration"),
    TargetSpec("nfl.vmax_p90", "observed high-speed peak (90th percentile)", "yd/s", "episodes", ("sustained_high",), "v_max", "p90",
               "sustained_high episodes", "forty.v_max_obs", "higher observed speed (not a maximum)"),
    TargetSpec("nfl.brake_decel", "mean deceleration while braking", "yd/s²", "episodes", ("braking",), "_neg_mean_accel", "median",
               "braking episodes from ≥ 4 yd/s with a drop ≥ 2 yd/s", "short_shuttle.decel_mean", "harder braking"),
    TargetSpec("nfl.brake_peak05", "peak 0.5-s deceleration", "yd/s²", "episodes", ("braking",), "_neg_peak_rate_05", "median",
               "braking episodes", "short_shuttle.decel_peak", "harder braking"),
    TargetSpec("nfl.brake_distance_matched", "braking distance from 4–7 yd/s", "yd", "episodes", ("braking",), "distance_yd", "median",
               "braking episodes starting at 4–7 yd/s (entry-speed matched)", "short_shuttle.braking_distance_yd", "longer stopping path"),
    TargetSpec("nfl.reaccel_mean", "re-acceleration after braking", "yd/s²", "episodes", ("reaccel_after_brake",), "mean_accel", "median",
               "accelerating episode starting ≤ 1 s after a braking episode", "short_shuttle.exit_accel", "harder re-acceleration"),
    TargetSpec("nfl.redirect_retention", "speed retained through a redirection", "fraction", "episodes", ("redirection",), "_retention", "median",
               "redirection episodes (|Δheading| ≥ 45°, valid heading, ≥ 3 yd/s)", "three_cone.transition_cost_index", "more speed kept"),
    TargetSpec("nfl.redirect_retention_90", "speed retained through a ≥ 90° redirection", "fraction", "episodes", ("redirection",), "_retention90", "median",
               "redirection episodes with |Δheading| ≥ 90°", "three_cone.transition_cost_index", "more speed kept"),
    TargetSpec("nfl.brake_to_burst", "Brake-to-Burst latency", "s", "pairs", (), "latency_s", "median",
               "braking end → start of the following re-acceleration (same player-play, ≤ 1 s)", "short_shuttle.brake_to_burst_s", "slower re-start"),
    TargetSpec("nfl.pair_tci", "Transition Cost Index of brake → re-acceleration", "fraction", "pairs", (), "tci_05", "median",
               "(entry − speed 0.5 s into the re-acceleration) / entry", "short_shuttle.transition_cost_index", "more speed lost"),
    TargetSpec("nfl.pair_tci_turn90", "Transition Cost Index with an observed ≥ 90° turn", "fraction", "pairs", (), "tci_05_turn90", "median",
               "pairs overlapping a redirection episode with |Δheading| ≥ 90°", "three_cone.transition_cost_index", "more speed lost"),
    TargetSpec("nfl.broad_accel_sust05", "season-average sustained acceleration (Milestone 05 primary, broad)", "yd/s²", "plays", (), "accel_sust05", "median",
               "every eligible player-play (no speed-state matching)", "", "harder acceleration"),
)
TARGET_IDS = tuple(t.target_id for t in TARGETS)


def target(tid: str) -> TargetSpec:
    return next(t for t in TARGETS if t.target_id == tid)


# ============================================================================== pairs
def brake_reaccel_pairs(episodes: pl.DataFrame, max_gap_s: float = 1.0, horizon_s: float = 0.5) -> pl.DataFrame:
    """One row per braking episode followed (≤ ``max_gap_s``) by a re-acceleration in the same player-play."""
    b = episodes.filter(pl.col("demand_id") == "braking").select(*PP_KEYS, "season", "position_group", pl.col("episode").alias("b_ep"),
                                                                  pl.col("t_start").alias("b_t"), pl.col("duration_s").alias("b_dur"),
                                                                  pl.col("v_start").alias("entry_speed"), pl.col("v_end").alias("v_min"),
                                                                  (-pl.col("mean_accel")).alias("brake_decel"))
    r = episodes.filter(pl.col("demand_id") == "reaccel_after_brake").select(*PP_KEYS, pl.col("t_start").alias("r_t"), pl.col("duration_s").alias("r_dur"),
                                                                             pl.col("v_start").alias("r_v0"), pl.col("v_max").alias("r_vmax"),
                                                                             pl.col("mean_accel").alias("reaccel_mean"))
    p = (b.join(r, on=PP_KEYS, how="inner").with_columns((pl.col("r_t") - (pl.col("b_t") + pl.col("b_dur"))).alias("latency_s"))
         .filter((pl.col("latency_s") >= -0.051) & (pl.col("latency_s") <= max_gap_s + 1e-6))
         .sort(*PP_KEYS, "b_ep", "latency_s").unique(subset=[*PP_KEYS, "b_ep"], keep="first"))
    p = p.with_columns((pl.col("r_v0") + pl.col("reaccel_mean") * pl.min_horizontal(pl.col("r_dur"), pl.lit(horizon_s))).alias("exit_speed_05"),
                       pl.col("latency_s").clip(lower_bound=0.0))
    p = p.with_columns(((pl.col("entry_speed") - pl.col("exit_speed_05")) / pl.col("entry_speed")).alias("tci_05"),
                       (pl.col("r_vmax") / pl.col("entry_speed")).alias("recovery_ratio"))
    turns = episodes.filter((pl.col("demand_id") == "redirection") & pl.col("heading_change_deg").is_not_null()).select(
        *PP_KEYS, pl.col("t_start").alias("h_t"), pl.col("duration_s").alias("h_dur"), pl.col("heading_change_deg").abs().alias("abs_turn"))
    ov = (p.select(*PP_KEYS, "b_ep", "b_t", "r_t", "r_dur").join(turns, on=PP_KEYS, how="inner")
          .filter((pl.col("h_t") <= pl.col("r_t") + pl.col("r_dur")) & (pl.col("h_t") + pl.col("h_dur") >= pl.col("b_t") - 0.5))
          .group_by(*PP_KEYS, "b_ep").agg(pl.col("abs_turn").max().alias("turn_angle_abs")))
    p = p.join(ov, on=[*PP_KEYS, "b_ep"], how="left").with_columns(
        pl.col("turn_angle_abs").is_not_null().alias("angle_observed"),
        pl.when(pl.col("turn_angle_abs") >= 90).then(pl.col("tci_05")).otherwise(None).alias("tci_05_turn90"))
    return p.drop("b_t", "r_t", "r_dur", "r_v0", "r_vmax")


# ============================================================================== athlete targets
def _episode_values(episodes: pl.DataFrame, pairs: pl.DataFrame, plays: pl.DataFrame | None, t: TargetSpec) -> pl.DataFrame:
    """(nfl_id, position_group, game_id, season, value) of every episode / pair / play contributing to ``t``."""
    if t.source == "pairs":
        d = pairs.select("nfl_id", "position_group", "game_id", "season", pl.col(t.expr).alias("value"))
    elif t.source == "plays":
        if plays is None or t.expr not in plays.columns:
            return pl.DataFrame(schema={"nfl_id": pl.Int64, "position_group": pl.String, "game_id": pl.Int64, "season": pl.Int64, "value": pl.Float64})
        d = plays.filter(pl.col("eligible")).select("nfl_id", "position_group", "game_id", "season", pl.col(t.expr).cast(pl.Float64).alias("value"))
    else:
        e = episodes.filter(pl.col("demand_id").is_in(list(t.demands)))
        if t.target_id == "nfl.brake_distance_matched":
            e = e.filter(pl.col("v_start").is_between(4.0, 7.0))
        expr = {"_neg_mean_accel": -pl.col("mean_accel"), "_neg_peak_rate_05": -pl.col("peak_rate_05"),
                "_retention": (pl.col("v_start") + pl.col("speed_change")) / pl.col("v_start"),
                "_retention90": pl.when(pl.col("heading_change_deg").abs() >= 90).then((pl.col("v_start") + pl.col("speed_change")) / pl.col("v_start"))
                }.get(t.expr, pl.col(t.expr) if t.expr in e.columns else pl.lit(None, dtype=pl.Float64))
        d = e.select("nfl_id", "position_group", "game_id", "season", expr.cast(pl.Float64).alias("value"))
    return d.filter(pl.col("value").is_not_null() & pl.col("value").is_finite())


def athlete_targets(episodes: pl.DataFrame, pairs: pl.DataFrame, plays: pl.DataFrame | None, c: M11Config,
                    specs: Sequence[TargetSpec] = TARGETS, by_season: bool = False) -> pl.DataFrame:
    """Long table: nfl_id, [season,] target_id, position_group, value, n_episodes, n_games, n_seasons, supported."""
    parts = []
    keys = ["nfl_id", "season"] if by_season else ["nfl_id"]
    for t in specs:
        v = _episode_values(episodes, pairs, plays, t)
        if v.is_empty():
            continue
        stat = pl.col("value").quantile(0.9, "linear") if t.stat == "p90" else pl.col("value").median()
        g = v.group_by(keys).agg(pl.col("position_group").first(), stat.alias("value"), pl.len().cast(pl.Int64).alias("n_episodes"),
                                 pl.col("game_id").n_unique().cast(pl.Int64).alias("n_games"), pl.col("season").n_unique().cast(pl.Int64).alias("n_seasons"))
        need = c.min_episodes_season if by_season else c.min_episodes_athlete
        parts.append(g.with_columns(pl.lit(t.target_id).alias("target_id"), (pl.col("n_episodes") >= need).alias("supported")))
    if not parts:
        return pl.DataFrame()
    return pl.concat(parts, how="diagonal_relaxed").sort(*keys, "target_id")


def target_reliability(episodes: pl.DataFrame, pairs: pl.DataFrame, plays: pl.DataFrame | None, c: M11Config,
                       specs: Sequence[TargetSpec] = TARGETS) -> pl.DataFrame:
    rows = []
    for t in specs:
        v = _episode_values(episodes, pairs, plays, t)
        r = split_half(v, "value", "game_id", min_per_half=max(2, c.min_episodes_athlete // 4)) if v.height else {"n_athletes": 0, "r_half": NAN, "reliability": NAN}
        rows.append({"target_id": t.target_id, "label": t.label, "unit": t.unit, **r, "n_episodes": v.height})
    return pl.DataFrame(rows)


def target_dictionary() -> pl.DataFrame:
    return pl.DataFrame([{"target_id": t.target_id, "label": t.label, "unit": t.unit, "source": t.source, "demands": ", ".join(t.demands),
                          "statistic": t.stat, "condition": t.where, "combine_match": t.combine_match or "none", "higher_is": t.higher_is} for t in TARGETS])


# ============================================================================== tracking scan (shared by the envelope builder and the window extractor)
def iter_post_snap(nb: NflOutcomeBuilder, eligible: pl.DataFrame, spec: SmoothingSpec, frame_tolerance_s: float, athlete_ids: Sequence[int] | None = None
                   ) -> Iterator[tuple[tuple[int, int], pl.DataFrame]]:
    """((season, batch), frame kinematics with ``t_rel`` and ``edge``) for every game batch: eligible player-plays only,
    post-snap frames only. Same reader, segmentation and smoothing as Milestones 05 and 09A."""
    anchors = nb.anchors()
    half = spec.window // 2
    for season, file in nb.kin.nfl_season_files().items():
        for bi, (lo, hi) in enumerate(nb.kin.game_batches(file), 1):
            snaps = anchors.filter(pl.col("game_id").is_between(lo, hi))
            want = eligible.filter(pl.col("game_id").is_between(lo, hi))
            if snaps.is_empty() or want.is_empty():
                continue
            raw = (nb.kin.load_nfl_raw(file, season, lo, hi).drop("s", "a", "dir", "o").with_columns([pl.col(k).cast(pl.Int64) for k in PP_KEYS])
                   .join(want, on=PP_KEYS, how="semi"))
            if athlete_ids is not None:
                raw = raw.filter(pl.col("nfl_id").is_in(list(athlete_ids)))
            if raw.is_empty():
                continue
            frames, _ = prepare_frames(raw, NFL, nb.kcfg)
            del raw
            kin = compute_kinematics(frames, nb.kcfg, spec)
            del frames
            kin = (kin.join(snaps.select(*PLAY_KEYS, "snap_ts").unique(subset=PLAY_KEYS), on=PLAY_KEYS, how="inner")
                   .with_columns(((pl.col("ts") - pl.col("snap_ts")).dt.total_microseconds().cast(pl.Float64) / 1e6).alias("t_rel"))
                   .filter(pl.col("t_rel") >= -frame_tolerance_s)
                   .with_columns(((pl.col("pos") < half) | (pl.col("seg_len") - 1 - pl.col("pos") < half)).alias("edge"))
                   .sort(*PP_KEYS, "seg_idx", "ts"))
            yield (int(season), bi), kin
            del kin
            gc.collect()


def _edges(c: M11Config) -> tuple[np.ndarray, np.ndarray]:
    se = np.arange(0.0, c.env_speed_max + c.env_speed_step / 2, c.env_speed_step)
    ae = np.arange(-c.env_accel_lim, c.env_accel_lim + c.env_accel_step / 2, c.env_accel_step)
    return se, ae


ENV_HIST_SCHEMA = {"nfl_id": pl.Int64, "game_id": pl.Int64, "season": pl.Int64, "speed_bin": pl.Int32, "accel_bin": pl.Int32, "n": pl.Int32}


class EnvelopeBuilder:
    """Athlete × game × speed bin × acceleration bin counts, resumable by (season, batch) checkpoint shards."""

    def __init__(self, cfg: Any, c: M11Config, oc: OutcomeConfig, shards_root: Any, mode: str, limit_games: int | None = None,
                 athlete_ids: Sequence[int] | None = None):
        self.c, self.oc = c, oc
        self.nb = NflOutcomeBuilder(cfg, oc, limit_games=limit_games, athlete_ids=athlete_ids)
        self.spec = SmoothingSpec(*oc.primary_smoothing)
        self.athlete_ids = athlete_ids
        se, ae = _edges(c)
        self.ckpt = ShardCheckpoint(shards_root, "envelope_hist", {"speed_edges": se.tolist(), "accel_edges": ae.tolist(), "smoothing": list(oc.primary_smoothing),
                                                                    "limit_games": limit_games, "athletes": None if athlete_ids is None else sorted(athlete_ids)}, mode)

    def compute(self, plays: pl.DataFrame) -> pl.DataFrame:
        se, ae = _edges(self.c)
        eligible = plays.filter(pl.col("eligible")).select(PP_KEYS)
        season_of = plays.select("game_id", "season").unique(subset="game_id")
        t0 = time.perf_counter()
        expected = []
        for key, kin in iter_post_snap(self.nb, eligible, self.spec, self.oc.frame_tolerance_s, self.athlete_ids):
            expected.append(key)
            if self.ckpt.has(key):
                log.info("[m11.envelope] batch %s restored from checkpoint", key)
                continue
            f = kin.filter(~pl.col("edge") & pl.col("speed").is_finite() & pl.col("a_tan").is_finite() & (pl.col("t_rel") >= 0))
            h = (f.select("nfl_id", "game_id",
                          ((pl.col("speed") / self.c.env_speed_step).floor().clip(0, se.size - 2)).cast(pl.Int32).alias("speed_bin"),
                          (((pl.col("a_tan") + self.c.env_accel_lim) / self.c.env_accel_step).floor().clip(0, ae.size - 2)).cast(pl.Int32).alias("accel_bin"))
                 .group_by("nfl_id", "game_id", "speed_bin", "accel_bin").agg(pl.len().cast(pl.Int32).alias("n"))
                 .join(season_of, on="game_id", how="left"))
            self.ckpt.save(key, h.select([pl.col(k).cast(t) for k, t in ENV_HIST_SCHEMA.items()]))
            log.info("[m11.envelope] season %s batch %d: %s cells (%.0fs)", key[0], key[1], f"{h.height:,}", time.perf_counter() - t0)
        self.ckpt.mark_complete(expected)
        out = self.ckpt.collect(expected)
        return out.select([pl.col(k).cast(t) for k, t in ENV_HIST_SCHEMA.items()]) if out.height else pl.DataFrame(schema=ENV_HIST_SCHEMA)


def _matrix(cells: pl.DataFrame, ns: int, na: int) -> np.ndarray:
    H = np.zeros((ns, na))
    if cells.height:
        np.add.at(H, (cells["speed_bin"].to_numpy(), cells["accel_bin"].to_numpy()), cells["n"].to_numpy().astype(float))
    return H


def envelope_table(hist: pl.DataFrame, c: M11Config, by: Sequence[str] = ("nfl_id",), n_boot: int | None = None, seed: int = 0) -> pl.DataFrame:
    """Envelope estimators per athlete (or athlete × season) with game-bootstrap intervals for the linear A0 / V0 and the
    envelope at band mid-speeds."""
    se, ae = _edges(c)
    ns, na = se.size - 1, ae.size - 1
    B = c.env_boot if n_boot is None else n_boot
    rows = []
    groups = hist.partition_by(list(by), as_dict=True)
    prog = Progress(len(groups), "envelopes", every_s=30)
    rng = np.random.default_rng(seed)
    for key, g in sorted(groups.items()):
        games = g.partition_by(["game_id"], as_dict=True)
        mats = {k[0]: _matrix(v, ns, na) for k, v in games.items()}
        H = sum(mats.values())
        s = ph.envelope_summary(H, se, ae, c.env_primary_quantile, c.env_min_frames_bin, c.env_min_bins, c.env_fit_vmin, c.bands)
        for q in c.env_quantiles:
            v_, aq, n_ = ph.envelope_points(H, se, ae, q, c.env_min_frames_bin, c.env_fit_vmin)
            lf = ph.fit_linear_envelope(v_, aq, n_) if v_.size >= c.env_min_bins else {"A0": NAN, "V0": NAN}
            s[f"lin_A0_q{int(q * 100)}"], s[f"lin_V0_q{int(q * 100)}"] = lf["A0"], lf["V0"]
        boot = {"lin_A0": [], "lin_V0": [], **{f"env_{lo:g}_{hi:g}": [] for lo, hi in c.bands}}
        if s["supported"] and len(mats) >= 5:
            keys = list(mats)
            stack = np.stack([mats[k] for k in keys])
            for _ in range(B):
                Hb = stack[rng.integers(0, len(keys), len(keys))].sum(0)
                v_, aq, n_ = ph.envelope_points(Hb, se, ae, c.env_primary_quantile, c.env_min_frames_bin, c.env_fit_vmin)
                if v_.size < c.env_min_bins:
                    continue
                lf = ph.fit_linear_envelope(v_, aq, n_)
                boot["lin_A0"].append(lf["A0"])
                boot["lin_V0"].append(lf["V0"])
                iso = ph.isotonic_decreasing(aq, n_)
                for lo, hi in c.bands:
                    mid = 0.5 * (lo + hi)
                    boot[f"env_{lo:g}_{hi:g}"].append(float(np.interp(mid, v_, iso)) if v_.min() <= mid <= v_.max() else NAN)
        for k, vals in boot.items():
            arr = np.asarray(vals, float)
            arr = arr[np.isfinite(arr)]
            s[f"{k}_lo"], s[f"{k}_hi"] = (float(np.quantile(arr, 0.025)), float(np.quantile(arr, 0.975))) if arr.size >= 20 else (NAN, NAN)
            s[f"{k}_se"] = float(arr.std()) if arr.size >= 20 else NAN
        rows.append({**dict(zip(by, key)), "n_games": len(mats), **s})
        prog.step()
    return pl.DataFrame(rows, infer_schema_length=None) if rows else pl.DataFrame()


def position_envelopes(hist: pl.DataFrame, positions: pl.DataFrame, c: M11Config) -> pl.DataFrame:
    """Pooled envelope points per position (frames of all athletes) for the distribution figure."""
    se, ae = _edges(c)
    ns, na = se.size - 1, ae.size - 1
    d = hist.join(positions.select("nfl_id", "position_group"), on="nfl_id", how="inner")
    rows = []
    for (pos,), g in sorted(d.partition_by(["position_group"], as_dict=True).items()):
        H = _matrix(g.group_by("speed_bin", "accel_bin").agg(pl.col("n").sum()), ns, na)
        for q in c.env_quantiles:
            v, aq, n = ph.envelope_points(H, se, ae, q, c.env_min_frames_bin, 0.0)
            rows += [{"position_group": pos, "quantile": q, "speed_mid": float(a), "accel_q": float(b), "n_frames": float(m)} for a, b, m in zip(v, aq, n)]
    return pl.DataFrame(rows)


def athlete_points(hist: pl.DataFrame, nfl_id: int, c: M11Config, q: float | None = None) -> pl.DataFrame:
    se, ae = _edges(c)
    H = _matrix(hist.filter(pl.col("nfl_id") == nfl_id).group_by("speed_bin", "accel_bin").agg(pl.col("n").sum()), se.size - 1, ae.size - 1)
    v, aq, n = ph.envelope_points(H, se, ae, q or c.env_primary_quantile, c.env_min_frames_bin, 0.0)
    return pl.DataFrame({"speed_mid": v, "accel_q": aq, "n_frames": n})


# ============================================================================== NFL windows for 11F
WINDOW_SCHEMA = {"window_id": pl.String, "nfl_id": pl.Int64, "demand_id": pl.String, "season": pl.Int64, "frame": pl.Int32, "speed": pl.Float32,
                 "a_tan": pl.Float32, "a_n": pl.Float32, "yaw_rate": pl.Float32, "dt": pl.Float32, "valid": pl.Boolean, "heading_valid": pl.Boolean}


def sample_episodes(episodes: pl.DataFrame, c: M11Config, seed: int) -> pl.DataFrame:
    """Deterministic capped sample: ≤ ``nfl_windows_per_athlete_demand`` episodes per athlete × demand (repeated_change
    excluded: a play-level aggregate), then ≤ ``nfl_windows_max`` overall."""
    e = episodes.filter(pl.col("demand_id") != "repeated_change").with_columns(
        pl.struct(*PP_KEYS, "episode").hash(seed).alias("_h"))
    e = e.sort("_h").with_columns(pl.int_range(pl.len()).over("nfl_id", "demand_id").alias("_r")).filter(pl.col("_r") < c.nfl_windows_per_athlete_demand)
    e = e.sort("_h").head(c.nfl_windows_max)
    return e.select(*PP_KEYS, "episode", "demand_id", "season", "t_start").with_columns(
        (pl.col("game_id").cast(pl.String) + "-" + pl.col("play_id").cast(pl.String) + "-" + pl.col("nfl_id").cast(pl.String) + "-"
         + pl.col("episode").cast(pl.String)).alias("window_id"))


class NflWindowExtractor:
    """Frame windows of ``seq_len`` frames starting 0.5 s before each sampled episode (resumable shards per batch)."""

    def __init__(self, cfg: Any, c: M11Config, oc: OutcomeConfig, shards_root: Any, mode: str, limit_games: int | None = None):
        self.c, self.oc = c, oc
        self.nb = NflOutcomeBuilder(cfg, oc, limit_games=limit_games)
        self.spec = SmoothingSpec(*oc.primary_smoothing)
        self.mode, self.root, self.limit_games = mode, shards_root, limit_games

    def compute(self, plays: pl.DataFrame, sample: pl.DataFrame) -> pl.DataFrame:
        ck = ShardCheckpoint(self.root, "nfl_windows", {"seq_len": self.c.seq_len, "n_sample": sample.height, "limit_games": self.limit_games,
                                                         "sample_hash": int(sample["window_id"].hash().sum()) if sample.height else 0}, self.mode)
        eligible = sample.select(PP_KEYS).unique()
        expected = []
        for key, kin in iter_post_snap(self.nb, eligible, self.spec, self.oc.frame_tolerance_s):
            expected.append(key)
            if ck.has(key):
                continue
            want = sample.filter(pl.col("game_id").is_in(kin["game_id"].unique().to_list()))
            rows = []
            for (g, p, i), f in kin.partition_by(PP_KEYS, as_dict=True).items():
                for w in want.filter((pl.col("game_id") == g) & (pl.col("play_id") == p) & (pl.col("nfl_id") == i)).iter_rows(named=True):
                    seg = f.filter(pl.col("t_rel") >= w["t_start"] - 0.5 - 1e-6).head(self.c.seq_len)
                    if seg.height < 5:
                        continue
                    sp = seg["speed"].cast(pl.Float64).to_numpy()
                    yaw = seg["yaw_rate"].cast(pl.Float64).to_numpy() if "yaw_rate" in seg.columns else np.full(seg.height, NAN)
                    an = seg["a_lat"].cast(pl.Float64).to_numpy() if "a_lat" in seg.columns else sp * np.radians(yaw)
                    tt = seg["t_rel"].to_numpy()
                    dt = np.r_[self.c.dt_s, np.diff(tt)]
                    same_seg = (seg["seg_idx"] == seg["seg_idx"][0]).to_numpy()
                    rows.append(pl.DataFrame({"window_id": w["window_id"], "nfl_id": int(i), "demand_id": w["demand_id"], "season": int(w["season"]),
                                              "frame": np.arange(seg.height, dtype=np.int32), "speed": sp.astype(np.float32),
                                              "a_tan": seg["a_tan"].cast(pl.Float32).to_numpy(), "a_n": an.astype(np.float32), "yaw_rate": yaw.astype(np.float32),
                                              "dt": dt.astype(np.float32), "valid": same_seg & ~seg["edge"].to_numpy(),
                                              "heading_valid": seg["heading_valid"].fill_null(False).to_numpy() if "heading_valid" in seg.columns else np.zeros(seg.height, bool)}))
            out = pl.concat(rows) if rows else pl.DataFrame(schema=WINDOW_SCHEMA)
            ck.save(key, out.select([pl.col(k).cast(t) for k, t in WINDOW_SCHEMA.items()]))
            log.info("[m11.windows] season %s batch %d: %d windows", key[0], key[1], len(rows))
        ck.mark_complete(expected)
        out = ck.collect(expected)
        return out if out.height else pl.DataFrame(schema=WINDOW_SCHEMA)


__all__ = ["ACCEL_DEMANDS", "ENV_HIST_SCHEMA", "EnvelopeBuilder", "NflWindowExtractor", "TARGETS", "TARGET_IDS", "TargetSpec", "WINDOW_SCHEMA",
           "athlete_points", "athlete_targets", "brake_reaccel_pairs", "envelope_table", "iter_post_snap", "position_envelopes", "sample_episodes",
           "target", "target_dictionary", "target_reliability"]
