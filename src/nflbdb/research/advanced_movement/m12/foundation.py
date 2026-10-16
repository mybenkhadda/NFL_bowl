"""12.0 — shared research foundation: reusable inputs, grids, cohorts, folds and feasibility for every Milestone 12 track.

Read only (nothing upstream is rebuilt)
    Milestone 11 (``discovery_v1``): ``m11_officials``, ``m11_athlete_features``, ``m11_attempt_features``, ``m11_attempt_index``,
    ``m11_nfl_targets``, ``m11_nfl_target_reliability``, ``m11_repeatability``, ``m11_folds``, ``m11_envelopes``,
    ``m11_envelope_hist`` (athlete × game × 0.5 yd/s speed bin × 0.25 yd/s² acceleration bin frame counts of eligible
    post-snap NFL frames — the NFL speed–acceleration observations of 12A / 12C / 12E; scanned lazily).
    Milestone 09: 09A episodes (game top speed), 09C stopwatch twins. Combine frames: ``phase_03/combine_frame_kinematics``
    through ``m11.sources.M11Sources`` (quality ``reject`` excluded, longest valid segment per attempt — Milestone 11 rule).

Written (``advanced_v1``)
    m12_combine_curves       forty attempt × speed bin: mean / max tangential acceleration on the RISING phase (onset → peak
                             speed), frames — the Combine a(v) function (no bin outside the observed rising phase)
    m12_combine_speed_time   forty attempt × time-from-onset (0.1 s grid ≤ 4 s): speed, observed only (no extrapolation)
    m12_combine_hist         attempt × speed bin × acceleration bin frame counts, every drill family, same edges as NFL
    m12_combine_quality      attempt: frames, median Δt, irregular-clock share, onset censored, peak at last frame
    m12_nfl_bin_support      athlete × speed bin: frames, games (from the NFL histograms)
    m12_game_top_speed       athlete × game: maximum episode speed (09A, edge episodes excluded)
    m12_cohort_registry      one row per athlete: position, cohort, linkage, repeats, support, fold, per-track eligibility
    m12_movement_grids       declared grids with units and support rules
    m12_shared_speed_support speed bin: share of linked athletes supported in each / both domains
    m12_fold_manifest        athlete folds (Milestone 11 folds reused, Milestone 06 folds inside them) + signature
    m12_experiment_availability  per track: eligible athletes, repeats, positions, linked, missing, scale, feasibility
    m12_foundation_manifest.json signatures (config, cohort, folds, grids, shared range) read by every later notebook
"""

from __future__ import annotations

import gc
from typing import Any, Sequence

import numpy as np
import polars as pl

from ....data.utils import get_logger, utc_now
from ..m11.combine_drills import longest_segment
from ..m11.sources import NB as NB11
from . import advanced_evaluation as ae
from .config import (FEAS_BLOCKED, FEAS_GATED, FEAS_LIMITED, FEAS_READY, M12Config, TRACKS, config_hash)

log = get_logger(__name__)
NAN = float("nan")
NB = {"12.0": "12_00_advanced_movement_research_foundation.ipynb", "12A": "12a_functional_movement_transfer.ipynb",
      "12B": "12b_measurement_error_and_latent_traits.ipynb", "12C": "12c_distributional_nfl_movement_capacity.ipynb",
      "12D": "12d_multiview_latent_athletic_traits.ipynb", "12E": "12e_conditional_optimal_transport_matching.ipynb",
      "12F": "12f_physics_informed_shared_movement_encoder.ipynb", "12G": "12g_advanced_movement_research_synthesis.ipynb"}
HIST_SCHEMA = {"nfl_id": pl.Int64, "game_id": pl.Int64, "season": pl.Int64, "speed_bin": pl.Int32, "accel_bin": pl.Int32, "n": pl.Int32}


# ============================================================================== read-only access
class M12Data:
    """Read-only accessor of every reused artifact. ``store``: ``artifacts.ResearchStore``. ``sources``: ``m11.sources.M11Sources``
    (only needed for Combine frames, 09A episodes and twins)."""

    def __init__(self, store: Any, c: M12Config, sources: Any | None = None, athlete_ids: Sequence[int] | None = None):
        self.store, self.c, self.S = store, c, sources
        self.ids = None if athlete_ids is None else sorted(int(i) for i in athlete_ids)

    def _m11(self, name: str, nb: str = "11.0") -> pl.DataFrame:
        d = self.store.discovery.read_frame(name, NB11[nb])
        return d.filter(pl.col("nfl_id").is_in(self.ids)) if self.ids is not None and "nfl_id" in d.columns else d

    def officials(self) -> pl.DataFrame:
        return self._m11("m11_officials")

    def athlete_features(self) -> pl.DataFrame:
        return self._m11("m11_athlete_features")

    def attempt_features(self) -> pl.DataFrame:
        return self._m11("m11_attempt_features")

    def attempt_index(self) -> pl.DataFrame:
        return self._m11("m11_attempt_index")

    def nfl_targets(self) -> pl.DataFrame:
        return self._m11("m11_nfl_targets")

    def nfl_target_reliability(self) -> pl.DataFrame:
        return self.store.discovery.read_frame("m11_nfl_target_reliability", NB11["11.0"])

    def repeatability(self) -> pl.DataFrame:
        return self.store.discovery.read_frame("m11_repeatability", NB11["11.0"])

    def folds(self) -> pl.DataFrame:
        return self._m11("m11_folds")

    def envelopes(self) -> pl.DataFrame:
        return self._m11("m11_envelopes", "11B")

    def hist(self) -> pl.LazyFrame:
        """Lazy NFL histogram scan (projection / predicate pushdown; never fully materialised by 12.0)."""
        lf = self.store.scan("m11_envelope_hist", NB11["11B"], discovery=True)
        return lf.filter(pl.col("nfl_id").is_in(self.ids)) if self.ids is not None else lf

    def hist_athletes(self, ids: Sequence[int], columns: Sequence[str] = tuple(HIST_SCHEMA)) -> pl.DataFrame:
        return self.hist().filter(pl.col("nfl_id").is_in(list(ids))).select(list(columns)).collect()

    def episodes(self) -> pl.DataFrame:
        if self.S is None:
            raise RuntimeError("M12Data needs m11 sources for the 09A episodes")
        e = self.S.episodes()
        return e.filter(pl.col("nfl_id").is_in(self.ids)) if self.ids is not None else e

    def twins(self) -> pl.DataFrame:
        if self.S is None:
            raise RuntimeError("M12Data needs m11 sources for the stopwatch twins")
        return self.S.twins()

    # ---- 12.0 outputs (later notebooks) -----------------------------------------------------------------------------
    def m12(self, name: str) -> pl.DataFrame:
        return self.store.read_frame(name, NB["12.0"])

    def manifest(self) -> dict[str, Any]:
        return self.store.read_json("m12_foundation_manifest", NB["12.0"])


# ============================================================================== Combine per-attempt tables
def _attempt_arrays(f: pl.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    g = lambda k: f[k].cast(pl.Float64).fill_null(np.nan).to_numpy() if k in f.columns else np.full(f.height, np.nan)
    return g("t_s"), g("speed"), g("a_tan")


def rising_phase(t: np.ndarray, v: np.ndarray, onset_speed: float) -> tuple[int | None, int]:
    """(onset index or None when the recording starts above ``onset_speed``, index of peak speed). The rising phase is
    onset → first frame of maximal speed; frames after the peak (deceleration through the finish) are excluded."""
    if v.size == 0 or not np.isfinite(v).any():
        return None, 0
    peak = int(np.nanargmax(v))
    if not np.isfinite(v[0]) or v[0] >= onset_speed:
        return None, peak
    idx = np.flatnonzero(v >= onset_speed)
    return (int(idx[0]) if idx.size else None), peak


CURVE_RULE = "rising_phase_interp_v2"     # D-12-1: interpolation at bin mid-speeds along the monotone rising phase
MAX_BRACKET_GAP = 1.0                     # yd/s: a bin mid-speed is supported only between frames ≤ this far apart in speed


def attempt_curve(t: np.ndarray, v: np.ndarray, a: np.ndarray, c: M12Config) -> list[dict[str, Any]]:
    """a(v) of one forty attempt at the 0.5 yd/s bin MID-SPEEDS, interpolated along the monotone part of the rising phase
    (onset → first frame of peak speed; frames that do not exceed the running maximum speed are dropped). At 10 Hz a
    sprinter gains ~0.5 yd/s per frame, so most bins contain ≤ 1 frame: frame binning (rule v1) left them empty although
    the attempt passed through them. A mid-speed is supported only between the attempt's first and last monotone frames
    and when its two bracketing frames are ≤ ``MAX_BRACKET_GAP`` apart in speed — nothing is extrapolated.
    ``n_frames`` = frames whose speed falls inside the bin (may be 0 for an interpolated bin)."""
    on, peak = rising_phase(t, v, c.onset_speed)
    lo = 0 if on is None else on
    if peak <= lo + 2:
        return []
    vs, as_ = v[lo: peak + 1], a[lo: peak + 1]
    ok = np.isfinite(vs) & np.isfinite(as_)
    vs, as_ = vs[ok], as_[ok]
    if vs.size < 3:
        return []
    keep = np.r_[True, vs[1:] > np.maximum.accumulate(vs)[:-1]]
    vm, am = vs[keep], as_[keep]
    se = ae.speed_edges(c)
    mids = ae.mids(se)
    counts = np.bincount(np.clip(np.floor(vs / c.speed_step).astype(int), 0, se.size - 2), minlength=se.size - 1)
    out = []
    for k, m in enumerate(mids):
        if m < vm[0] or m > vm[-1]:
            continue
        j = int(np.searchsorted(vm, m))
        if 0 < j < vm.size and vm[j] - vm[j - 1] > MAX_BRACKET_GAP:
            continue
        a_mid = float(np.interp(m, vm, am))
        lo_k, hi_k = se[k], se[k + 1]
        inbin = (vs >= lo_k) & (vs < hi_k)
        out.append({"speed_bin": int(k), "speed_mid": float(m), "a_mean": a_mid, "a_max": float(as_[inbin].max()) if inbin.any() else a_mid,
                    "n_frames": int(counts[k]), "onset_censored": on is None, "curve_rule": CURVE_RULE})
    return out


def attempt_speed_time(t: np.ndarray, v: np.ndarray, c: M12Config) -> np.ndarray | None:
    """Speed on the time-from-onset grid; NaN beyond the recording; None when the onset is not recorded."""
    on, _ = rising_phase(t, v, c.onset_speed)
    if on is None:
        return None
    tt = t[on:] - t[on]
    vv = v[on:]
    ok = np.isfinite(tt) & np.isfinite(vv)
    if ok.sum() < 5:
        return None
    g = ae.time_grid(c)
    out = np.interp(g, tt[ok], vv[ok])
    out[g > tt[ok].max() + 1e-9] = np.nan
    return out


def attempt_hist(v: np.ndarray, a: np.ndarray, c: M12Config) -> pl.DataFrame:
    se, ab = ae.speed_edges(c), ae.accel_edges(c)
    ok = np.isfinite(v) & np.isfinite(a)
    sb = np.clip(np.floor(v[ok] / c.speed_step).astype(int), 0, se.size - 2)
    abn = np.clip(np.floor((a[ok] + c.accel_lim) / c.accel_step).astype(int), 0, ab.size - 2)
    if sb.size == 0:
        return pl.DataFrame(schema={"speed_bin": pl.Int32, "accel_bin": pl.Int32, "n": pl.Int32})
    d = pl.DataFrame({"speed_bin": sb.astype(np.int32), "accel_bin": abn.astype(np.int32)})
    return d.group_by("speed_bin", "accel_bin").agg(pl.len().cast(pl.Int32).alias("n"))


def combine_tables(sources: Any, c: M12Config, families: Sequence[str], athlete_ids: Sequence[int] | None = None) -> dict[str, pl.DataFrame]:
    """One pass over the Combine frames, family by family (bounded memory)."""
    curves, st, hists, qual = [], [], [], []
    g = ae.time_grid(c)
    for fam in families:
        fr = sources.combine_frames([fam])
        if athlete_ids is not None:
            fr = fr.filter(pl.col("nfl_id").is_in(list(athlete_ids)))
        if fr.is_empty():
            continue
        order = (fr.group_by("seq_id").agg(pl.col("nfl_id").first(), pl.col("drill").first(), pl.col("ts").min().alias("_s"))
                 .with_columns(pl.col("_s").rank("ordinal").over("nfl_id", "drill").cast(pl.Int64).alias("attempt_order")))
        meta = {r["seq_id"]: r for r in order.iter_rows(named=True)}
        for (sid,), grp in fr.partition_by(["seq_id"], as_dict=True, maintain_order=True).items():
            f = longest_segment(grp)
            if f.height < 6:
                continue
            t, v, a = _attempt_arrays(f)
            m = meta[sid]
            base = {"seq_id": sid, "nfl_id": int(m["nfl_id"]), "family": fam, "drill": m["drill"], "attempt_order": int(m["attempt_order"])}
            dt = np.diff(t)
            on, peak = rising_phase(t, v, c.onset_speed)
            qual.append({**base, "n_frames": f.height, "median_dt": float(np.nanmedian(dt)) if dt.size else NAN,
                         "irregular_share": float(np.mean(np.abs(dt - c.dt_s) > 0.025)) if dt.size else NAN,
                         "onset_censored": on is None, "peak_at_last_frame": bool(peak >= f.height - 1),
                         "v_max": float(np.nanmax(v)) if np.isfinite(v).any() else NAN, "duration_s": float(t[-1] - t[0]) if t.size > 1 else NAN})
            h = attempt_hist(v, a, c)
            if h.height:
                hists.append(h.with_columns(pl.lit(sid).alias("seq_id"), pl.lit(int(m["nfl_id"])).cast(pl.Int64).alias("nfl_id"), pl.lit(fam).alias("family")))
            if fam == "forty":
                curves += [{**base, **r} for r in attempt_curve(t, v, a, c)]
                s = attempt_speed_time(t, v, c)
                if s is not None:
                    st.append(pl.DataFrame({"seq_id": sid, "nfl_id": int(m["nfl_id"]), "attempt_order": int(m["attempt_order"]), "t_rel": g, "speed": s})
                              .filter(pl.col("speed").is_not_nan()))
        del fr
        gc.collect()
        log.info("[m12.foundation] family %s done", fam)
    return {"m12_combine_curves": pl.DataFrame(curves, infer_schema_length=None) if curves else pl.DataFrame(),
            "m12_combine_speed_time": pl.concat(st) if st else pl.DataFrame(),
            "m12_combine_hist": pl.concat(hists, how="diagonal_relaxed").select("seq_id", "nfl_id", "family", "speed_bin", "accel_bin", "n") if hists else pl.DataFrame(),
            "m12_combine_quality": pl.DataFrame(qual, infer_schema_length=None) if qual else pl.DataFrame()}


# ============================================================================== NFL support (lazy)
def nfl_bin_support(hist: pl.LazyFrame) -> pl.DataFrame:
    return (hist.group_by("nfl_id", "speed_bin").agg(pl.col("n").sum().cast(pl.Int64).alias("n_frames"), pl.col("game_id").n_unique().alias("n_games"))
            .collect(engine="streaming").sort("nfl_id", "speed_bin"))


def nfl_athlete_summary(hist: pl.LazyFrame) -> pl.DataFrame:
    return (hist.group_by("nfl_id").agg(pl.col("n").sum().cast(pl.Int64).alias("nfl_frames"), pl.col("game_id").n_unique().alias("nfl_games"),
                                        pl.col("season").n_unique().alias("nfl_seasons")).collect(engine="streaming"))


def game_top_speed(episodes: pl.DataFrame) -> pl.DataFrame:
    """Per athlete × game: the maximum episode speed (09A episodes, every demand, edge episodes already excluded). A game
    maximum is an observed peak under that game's opportunities — not a capacity."""
    return (episodes.filter(pl.col("v_max").is_finite()).group_by("nfl_id", "game_id")
            .agg(pl.col("season").first(), pl.col("position_group").first(), pl.col("v_max").max().alias("top_speed"), pl.len().alias("n_episodes"))
            .sort("nfl_id", "game_id"))


# ============================================================================== shared speed support
def shared_speed_support(nfl_sup: pl.DataFrame, comb_curves: pl.DataFrame, linked: Sequence[int], c: M12Config) -> tuple[pl.DataFrame, dict[str, Any]]:
    """Share of LINKED athletes supporting each speed bin in the NFL (≥ ``min_frames_bin_nfl`` frames over ≥ 2 games) and in
    the Combine (≥ ``min_frames_bin_combine`` rising-phase frames in ≥ 1 attempt), and in both. The shared range is the
    longest contiguous run of bins whose BOTH-domain share ≥ ``shared_support_share``. Support decides; values never do."""
    se = ae.speed_edges(c)
    ids = set(int(i) for i in linked)
    n = max(1, len(ids))
    ns = nfl_sup.filter(pl.col("nfl_id").is_in(list(ids)) & (pl.col("n_frames") >= c.min_frames_bin_nfl) & (pl.col("n_games") >= 2)).select("nfl_id", "speed_bin")
    cs = (comb_curves.filter(pl.col("nfl_id").is_in(list(ids)) & (pl.col("n_frames") >= c.min_frames_bin_combine)).select("nfl_id", "speed_bin").unique()
          if comb_curves.height else pl.DataFrame(schema={"nfl_id": pl.Int64, "speed_bin": pl.Int32}))
    rows = []
    for k in range(se.size - 1):
        a = set(ns.filter(pl.col("speed_bin") == k)["nfl_id"].to_list())
        b = set(cs.filter(pl.col("speed_bin") == k)["nfl_id"].to_list())
        rows.append({"speed_bin": k, "speed_lo": float(se[k]), "speed_hi": float(se[k + 1]), "share_nfl": len(a) / n, "share_combine": len(b) / n,
                     "share_both": len(a & b) / n, "n_both": len(a & b)})
    df = pl.DataFrame(rows)
    ok = (df["share_both"] >= c.shared_support_share).to_numpy()
    best, cur, start = (0, -1, -1), 0, 0
    for k, flag in enumerate(ok):
        if flag:
            if cur == 0:
                start = k
            cur += 1
            if cur > best[0]:
                best = (cur, start, k)
        else:
            cur = 0
    if best[0] == 0:
        return df, {"bins": [], "lo": NAN, "hi": NAN, "rule": f"no bin reaches share_both ≥ {c.shared_support_share}"}
    bins = list(range(best[1], best[2] + 1))
    return df, {"bins": bins, "lo": float(se[bins[0]]), "hi": float(se[bins[-1] + 1]),
                "rule": f"longest contiguous run of bins with share_both ≥ {c.shared_support_share} among {len(ids)} linked athletes"}


# ============================================================================== registry and manifests
def movement_grids(c: M12Config, shared: dict[str, Any]) -> pl.DataFrame:
    se, ab = ae.speed_edges(c), ae.accel_edges(c)
    rows = [
        {"grid_id": "speed", "variable": "speed", "unit": "yd/s", "lo": float(se[0]), "hi": float(se[-1]), "step": c.speed_step, "n_points": se.size - 1,
         "support_rule": f"NFL bin ≥ {c.min_frames_bin_nfl} frames over ≥ 2 games; Combine bin mid-speed inside the monotone rising phase, "
                         f"bracketing frames ≤ {MAX_BRACKET_GAP} yd/s apart ({CURVE_RULE})"},
        {"grid_id": "speed_shared", "variable": "speed", "unit": "yd/s", "lo": shared.get("lo"), "hi": shared.get("hi"), "step": c.speed_step,
         "n_points": len(shared.get("bins", [])), "support_rule": shared.get("rule", "")},
        {"grid_id": "tangential_accel", "variable": "a_tan", "unit": "yd/s²", "lo": float(ab[0]), "hi": float(ab[-1]), "step": c.accel_step, "n_points": ab.size - 1,
         "support_rule": "same edges as m11_envelope_hist; values outside are clipped into the edge bins (as in Milestone 11)"},
        {"grid_id": "time_from_onset", "variable": "t", "unit": "s", "lo": 0.0, "hi": c.time_max, "step": c.time_step, "n_points": ae.time_grid(c).size,
         "support_rule": f"forty only, onset recorded (first speed < {c.onset_speed} yd/s); NaN after the recording ends"},
        {"grid_id": "curvature", "variable": "a_n / v²", "unit": "1/yd", "lo": None, "hi": None, "step": None, "n_points": 0,
         "support_rule": "NOT gridded: lateral acceleration is not in the NFL histograms; used only by 12F windows where heading is valid"},
        {"grid_id": "heading_change", "variable": "Δheading", "unit": "deg", "lo": 0.0, "hi": 180.0, "step": 45.0, "n_points": 4,
         "support_rule": "09A redirection episodes with valid heading; Combine transitions with turn speed ≥ 1 yd/s (Milestone 11)"},
        {"grid_id": "distance", "variable": "path distance", "unit": "yd", "lo": 0.0, "hi": 40.0, "step": 5.0, "n_points": 8,
         "support_rule": "forty only, from onset; reported for orientation (12A phase analysis)"}]
    return pl.DataFrame(rows, infer_schema_length=None)


def cohort_registry(off: pl.DataFrame, folds: pl.DataFrame, attempts_idx: pl.DataFrame, curves: pl.DataFrame, nfl_sum: pl.DataFrame,
                    nfl_sup: pl.DataFrame, envelopes: pl.DataFrame, gts: pl.DataFrame, shared: dict[str, Any], c: M12Config) -> pl.DataFrame:
    """One row per athlete of the Combine registry (+ NFL-only athletes of the histograms) with per-track eligibility flags.
    Flags use support and linkage only (no outcome value)."""
    fam = (attempts_idx.group_by("nfl_id").agg(pl.col("family").n_unique().alias("n_families"),
                                               (pl.col("family") == "forty").sum().cast(pl.Int64).alias("n_forty_attempts")))
    bins = shared.get("bins", [])
    cs = (curves.filter(pl.col("speed_bin").is_in(bins) & (pl.col("n_frames") >= c.min_frames_bin_combine)).group_by("nfl_id")
          .agg(pl.col("speed_bin").n_unique().alias("combine_shared_bins")) if curves.height and bins else pl.DataFrame(schema={"nfl_id": pl.Int64, "combine_shared_bins": pl.UInt32}))
    ns = (nfl_sup.filter(pl.col("speed_bin").is_in(bins) & (pl.col("n_frames") >= c.min_frames_bin_nfl) & (pl.col("n_games") >= 2)).group_by("nfl_id")
          .agg(pl.col("speed_bin").n_unique().alias("nfl_shared_bins")) if bins else pl.DataFrame(schema={"nfl_id": pl.Int64, "nfl_shared_bins": pl.UInt32}))
    g = gts.group_by("nfl_id").agg(pl.len().alias("games_with_top_speed"))
    env = envelopes.select("nfl_id", pl.col("supported").alias("has_envelope")) if envelopes.height else pl.DataFrame(schema={"nfl_id": pl.Int64, "has_envelope": pl.Boolean})
    ids = pl.concat([off.select("nfl_id"), nfl_sum.select("nfl_id")]).unique()
    r = (ids.join(off.select("nfl_id", "position_group", "draft_year", "forty", "ten", "height", "weight"), on="nfl_id", how="left")
         .join(fam, on="nfl_id", how="left").join(nfl_sum, on="nfl_id", how="left").join(cs, on="nfl_id", how="left").join(ns, on="nfl_id", how="left")
         .join(g, on="nfl_id", how="left").join(env, on="nfl_id", how="left").join(folds.select("nfl_id", "fold", "fold_source"), on="nfl_id", how="left"))
    r = r.with_columns([pl.col(k).fill_null(0) for k in ("n_families", "n_forty_attempts", "nfl_frames", "nfl_games", "nfl_seasons", "combine_shared_bins",
                                                          "nfl_shared_bins", "games_with_top_speed")]).with_columns(pl.col("has_envelope").fill_null(False))
    nb = max(1, len(bins))
    r = r.with_columns(
        pl.col("position_group").is_not_null().alias("in_combine"),
        (pl.col("nfl_games") > 0).alias("in_nfl"),
        (pl.col("n_forty_attempts") >= 1).alias("has_forty"),
        (pl.col("n_forty_attempts") >= 2).alias("has_forty_repeat"),
        pl.col("fold").is_not_null().alias("has_fold"))
    r = r.with_columns((pl.col("in_combine") & pl.col("in_nfl") & pl.col("has_fold")).alias("linked"))
    return r.with_columns(
        (pl.col("in_nfl") & (pl.col("nfl_games") >= c.min_games_athlete)).alias("elig_12C_nfl"),
        (pl.col("linked") & (pl.col("nfl_games") >= c.min_games_athlete) & pl.col("has_forty")).alias("elig_12C_transfer"),
        (pl.col("linked") & pl.col("has_forty_repeat")).alias("elig_12B"),
        (pl.col("linked") & (pl.col("combine_shared_bins") >= 0.6 * nb) & (pl.col("nfl_shared_bins") >= 0.6 * nb)).alias("elig_12A"),
        (pl.col("in_combine") & (pl.col("n_families") >= 2)).alias("elig_12D"),
        (pl.col("linked") & pl.col("has_forty") & (pl.col("nfl_games") >= c.min_games_athlete)).alias("elig_12E"),
        ((pl.col("n_families") >= 1) | pl.col("in_nfl")).alias("elig_12F")).sort("nfl_id")


def availability(reg: pl.DataFrame, shared: dict[str, Any], torch_ok: bool, c: M12Config) -> pl.DataFrame:
    """Per-track feasibility table (brief §6)."""
    def cnt(flag: str) -> int:
        return int(reg[flag].sum()) if flag in reg.columns else 0

    def pos(flag: str) -> str:
        p = reg.filter(pl.col(flag)).group_by("position_group").len().filter(pl.col("len") >= c.min_athletes)
        return ", ".join(sorted(x for x in p["position_group"].drop_nulls().to_list()))

    nbins = len(shared.get("bins", []))
    rows = [
        ("12C", "elig_12C_transfer", int(reg.filter(pl.col("elig_12C_nfl"))["nfl_games"].sum()), "NFL games (distribution splits)", "m11_envelope_hist, m11_envelopes, 09A episodes",
         "quantile curves, game-split reliability, ladder M0–M6", "play context (down, alignment) is not in the histograms", "≈ athletes × 23 × 64 histogram cells"),
        ("12B", "elig_12B", int(reg.filter(pl.col("elig_12B"))["n_forty_attempts"].sum()), "forty attempts", "m11_attempt_features, m11_repeatability",
         "variance components, RC, SIMEX, EM, synthetic grid", "only 1–3 attempts per athlete", "small"),
        ("12A", "elig_12A", int(reg.filter(pl.col("elig_12A"))["n_forty_attempts"].sum()), "forty attempts", "m12_combine_curves, m11_envelope_hist",
         "splines, FPCA, function-on-function, elastic alignment", "" if nbins else "NO shared speed range", "small"),
        ("12D", "elig_12D", int(reg.filter(pl.col("elig_12D"))["n_families"].sum()), "drill families", "m11_attempt_features, m11_athlete_features",
         "missing-data factor models", "drill participation depends on position", "small"),
        ("12E", "elig_12E", int(reg.filter(pl.col("elig_12E"))["nfl_games"].sum()), "NFL games", "m12_combine_hist, m11_envelope_hist",
         "Sinkhorn / sliced / conditional OT, bootstrap", "Combine forty has few frames per speed bin", "athletes² × bins² for Sinkhorn (bounded)"),
        ("12F", "elig_12F", int(reg["n_forty_attempts"].sum()), "Combine attempts", "m11 representation code; NFL windows NOT yet extracted",
         "one tracking scan for windows + encoder training", "" if torch_ok else "PyTorch not importable", "≤ 200k windows × 48 × 6 float32 ≈ 0.23 GB"),
    ]
    out = []
    for tid, flag, reps, rep_unit, existing, new, missing, scale in rows:
        n = cnt(flag)
        if tid == "12F" and not torch_ok:
            feas = FEAS_BLOCKED
        elif tid == "12A" and not nbins:
            feas = FEAS_BLOCKED
        elif n < c.min_athletes:
            feas = FEAS_LIMITED
        elif tid in ("12A", "12F"):
            feas = FEAS_GATED if tid == "12A" else FEAS_READY
        else:
            feas = FEAS_READY
        out.append({"track": tid, "eligible_athletes": n, "repeat_observations": reps, "repeat_unit": rep_unit, "supported_positions": pos(flag),
                    "linked_athletes": int(reg.filter(pl.col(flag) & pl.col("linked")).height), "missing_variables": missing,
                    "expected_scale": scale, "existing_artifacts": existing, "new_computation": new, "feasibility": feas})
    return pl.DataFrame(out)


def dependency_manifest() -> dict[str, Any]:
    return {"tracks": {t.track: {"notebook": t.notebook, "depends_on": list(t.depends_on), "primary_outcome": t.primary_outcome,
                                 "primary_comparison": t.primary_comparison, "evaluation_target": t.evaluation_target} for t in TRACKS},
            "recommended_order": ["12.0", "12C", "12B", "12A", "12D", "12E", "12F", "12G"],
            "note": "12F may be deferred; 12G reads whatever is executed and labels missing tracks as execution_pending."}


def foundation_manifest(c: M12Config, reg: pl.DataFrame, folds: pl.DataFrame, grids: pl.DataFrame, shared: dict[str, Any]) -> dict[str, Any]:
    linked = reg.filter(pl.col("linked"))["nfl_id"].to_list()
    return {"config_hash": c.hash(), "cohort_signature": ae.cohort_signature(linked), "n_linked": len(linked),
            "fold_signature": ae.fold_signature(folds), "grid_signature": config_hash(grids.rows()), "shared_speed": shared,
            "speed_edges": ae.speed_edges(c).tolist(), "accel_edges": ae.accel_edges(c).tolist(), "time_grid": ae.time_grid(c).tolist(), "written_at": utc_now()}


def check_foundation(store: Any, c: M12Config) -> dict[str, Any]:
    """Every later notebook calls this: the foundation must exist in this run mode and carry the current definitions.
    A configuration change in a GRID or SUPPORT parameter makes the foundation stale (re-run 12.0)."""
    m = store.read_json("m12_foundation_manifest", NB["12.0"])
    grid_keys = ("speed_step", "speed_max", "accel_step", "accel_lim", "time_step", "time_max", "onset_speed", "min_frames_bin_nfl",
                 "min_frames_bin_combine", "min_games_athlete", "shared_support_share")
    if m.get("speed_edges") != ae.speed_edges(c).tolist() or m.get("accel_edges") != ae.accel_edges(c).tolist():
        raise RuntimeError("12.0 foundation was built with other grids: re-run 12.0 with the current M12Config")
    m["grid_keys_checked"] = list(grid_keys)
    return m


__all__ = ["CURVE_RULE", "HIST_SCHEMA", "MAX_BRACKET_GAP", "M12Data", "NB", "attempt_curve", "attempt_hist", "attempt_speed_time", "availability", "check_foundation", "cohort_registry",
           "combine_tables", "dependency_manifest", "foundation_manifest", "game_top_speed", "movement_grids", "nfl_athlete_summary", "nfl_bin_support",
           "rising_phase", "shared_speed_support"]
