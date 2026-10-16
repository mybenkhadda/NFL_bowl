"""Per-sequence summaries and deterministic choice of example athletes for the EDA notebook.

Examples are chosen by rule (quantiles, calipers, seeded draws) so the same athletes appear on every rerun and
nobody hand-picks a flattering case. Every example is DESCRIPTIVE: it is shown, not used to fit or select anything.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import polars as pl

from ..contracts import TableContract
from ..sprint_rescue.stats import seed_for
from .sequences import SequenceSet

S, I, F, B = pl.String(), pl.Int64(), pl.Float64(), pl.Boolean()
SUMMARY = TableContract(
    "sequence_summary", "Per-sequence descriptive statistics of the eligible primary-drill sequences.",
    {"seq_id": S, "nfl_id": I, "position_group": S, "draft_year": I, "attempt_order": I, "n_frames": I, "duration_s": F,
     "path_yd": F, "first_speed": F, "peak_speed": F, "frames_after_peak": I, "peak_a_tan": F, "max_abs_lat_dev": F,
     "pos_resid_rms": F, "a_tan_roughness": F, "share_boundary": F}, key=("seq_id",), area="sequence",
    docs={"first_speed": "speed on frame 0 [yd/s] — above 0 means the recording began after the athlete started",
          "frames_after_peak": "0 = the recording ends at the speed peak (no plateau recorded)",
          "pos_resid_rms": "RMS distance between raw and smoothed position [yd] (tracking noise)",
          "a_tan_roughness": "SD of the frame-to-frame change of tangential acceleration [yd/s²]",
          "share_boundary": "share of frames inside the smoothing half-window of a segment edge"})
DECISIONS = TableContract(
    "preprocessing_decisions", "Preprocessing decisions of notebook 01 with the evidence each one rests on.",
    {"decision": S, "options": S, "evidence": S, "choice": S, "status": S, "rationale": S}, key=("decision",),
    area="decision", docs={"status": "PENDING REVIEW until the user records a choice after reading the executed notebook"})


def sequence_summary(ss: SequenceSet) -> pl.DataFrame:
    rows = []
    meta = ss.meta.select("seq_id", "nfl_id", "position_group", "draft_year", "attempt_order").to_dicts()
    for i, m in enumerate(meta):
        v, a = ss.series(i, "speed"), ss.series(i, "a_tan")
        k = int(ss.lengths[i])
        ipk = int(np.nanargmax(v)) if np.isfinite(v).any() else 0
        resid = ss.series(i, "pos_resid") if "pos_resid" in ss.channels else np.full(k, np.nan)
        lat = ss.series(i, "lat_dev") if "lat_dev" in ss.channels else np.full(k, np.nan)
        da = np.diff(a[np.isfinite(a)])
        rows.append({**m, "n_frames": k, "duration_s": float(ss.t[i, k - 1] - ss.t[i, 0]),
                     "path_yd": float(np.nanmax(ss.series(i, "dist"))) if "dist" in ss.channels else None,
                     "first_speed": float(v[0]), "peak_speed": float(np.nanmax(v)), "frames_after_peak": k - 1 - ipk,
                     "peak_a_tan": float(np.nanmax(a)) if np.isfinite(a).any() else None,
                     "max_abs_lat_dev": float(np.nanmax(np.abs(lat))) if np.isfinite(lat).any() else None,
                     "pos_resid_rms": float(np.sqrt(np.nanmean(resid ** 2))) if np.isfinite(resid).any() else None,
                     "a_tan_roughness": float(da.std(ddof=1)) if da.size > 2 else None,
                     "share_boundary": float(ss.boundary[i, :k].mean())})
    return pl.DataFrame(rows, schema=SUMMARY.columns)


def representative_sequences(summary: pl.DataFrame, by: str = "peak_speed", quantiles: Sequence[float] = (0.1, 0.5, 0.9)
                             ) -> list[str]:
    """The sequence closest to each quantile of ``by`` (deterministic)."""
    s = summary.filter(pl.col(by).is_not_null()).sort(by, "seq_id")
    if not s.height:
        return []
    return [s["seq_id"][int(round(q * (s.height - 1)))] for q in quantiles]


def repeat_athletes(summary: pl.DataFrame, k: int, seed: int, per_position: bool = True) -> list[int]:
    """``k`` athletes with at least two sequences; with ``per_position`` they are spread over position groups."""
    rep = (summary.group_by("nfl_id").agg(pl.len().alias("n"), pl.col("position_group").first())
           .filter(pl.col("n") >= 2).sort("nfl_id"))
    if not rep.height:
        return []
    rng = np.random.default_rng(seed_for(seed, "repeat_examples"))
    if not per_position:
        return sorted(rng.choice(rep["nfl_id"].to_numpy(), size=min(k, rep.height), replace=False).tolist())
    out: list[int] = []
    groups = [rng.permutation(rep.filter(pl.col("position_group") == p)["nfl_id"].to_numpy()).tolist()
              for p in sorted(rep["position_group"].drop_nulls().unique().to_list())]
    while len(out) < k and any(groups):
        for g in groups:
            if g and len(out) < k:
                out.append(int(g.pop()))
    return out


def similar_forty_pairs(summary: pl.DataFrame, athletes: pl.DataFrame, k: int, tol_s: float = 0.01,
                        contrast: str = "first_speed") -> pl.DataFrame:
    """Pairs of athletes of the SAME position whose official forty differs by ≤ ``tol_s`` — "the same stopwatch".
    One row per pair, first-attempt sequences, ordered by the largest difference in peak tangential acceleration
    among sequences with a comparable recording start (difference in ``contrast`` ≤ 0.5 yd/s), each athlete once."""
    first = (summary.sort("nfl_id", "attempt_order").group_by("nfl_id", maintain_order=True).first()
             .join(athletes.select("nfl_id", "display_name", "cmb_forty"), on="nfl_id", how="inner")
             .filter(pl.col("cmb_forty").is_not_null() & pl.col("peak_a_tan").is_not_null()).sort("nfl_id"))
    r = first.to_dicts()
    cand = []
    for i in range(len(r)):
        for j in range(i + 1, len(r)):
            a, b = r[i], r[j]
            if (a["position_group"] == b["position_group"] and abs(a["cmb_forty"] - b["cmb_forty"]) <= tol_s + 1e-12
                    and abs((a[contrast] or 0.0) - (b[contrast] or 0.0)) <= 0.5):
                cand.append((abs(a["peak_a_tan"] - b["peak_a_tan"]), i, j))
    used: set[int] = set()
    rows = []
    for d, i, j in sorted(cand, key=lambda c: (-c[0], c[1], c[2])):
        if i in used or j in used or len(rows) >= k:
            continue
        used |= {i, j}
        a, b = r[i], r[j]
        rows.append({"position_group": a["position_group"], "seq_id_a": a["seq_id"], "name_a": a["display_name"],
                     "forty_a": a["cmb_forty"], "seq_id_b": b["seq_id"], "name_b": b["display_name"],
                     "forty_b": b["cmb_forty"], "d_peak_a_tan": float(d)})
    return pl.DataFrame(rows, schema={"position_group": S, "seq_id_a": S, "name_a": S, "forty_a": F, "seq_id_b": S,
                                      "name_b": S, "forty_b": F, "d_peak_a_tan": F})


def decisions_table(evidence: dict[str, str], choices: dict[str, str | None], rationale: dict[str, str]) -> pl.DataFrame:
    """One row per preprocessing decision. A decision without a recorded choice stays PENDING REVIEW."""
    options = {"alignment": "elapsed_time | normalized_time | distance | speed_phase",
               "input_channels": "subset of the channel registry",
               "quality_scenario": "S0_all | S1_structural | S2_signal | S3_no_boundary",
               "boundary_frames": "keep | mask in losses | drop"}
    rows = [{"decision": d, "options": o, "evidence": evidence.get(d, ""), "choice": choices.get(d) or "",
             "status": "DECIDED" if choices.get(d) else "PENDING REVIEW", "rationale": rationale.get(d, "")}
            for d, o in options.items()]
    return pl.DataFrame(rows, schema=DECISIONS.columns)
