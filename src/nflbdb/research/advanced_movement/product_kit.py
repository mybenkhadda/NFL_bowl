"""Milestone 10 — product kit of the Combine Intelligence Lab: the pieces every final view shares.

Standard library only (plain dicts in, plain dicts / text out), so the Streamlit application and the Milestone 10
notebooks run exactly the same code on the Milestone 09 application mart.

    SPRINT_PHASES           five presentation phases of a forty, each a fixed set of the 09B curve features
    phase_decomposition     per phase: observed values, difference, interval, verdict — or "not identified"
    evidence_label          per feature: repeatability · redundancy · NFL transfer · coverage · descriptive status
    attempt_stability       per athlete × feature: repeat-attempt range and standard error (or "single attempt")
    sync_series             one recorded attempt on a common time or distance grid, NEVER beyond the recording
    NO_DATA                 the declared message of every missing-data state (never an invented value)

Nothing here recomputes a scientific quantity: differences, intervals and verdicts are the 09C values as stored in
``twin_differences``; evidence statuses are the 09B / 08B.4 values stored in ``feature_evidence``. The phase grouping is
a presentation choice declared below; the 09B phase of every feature is kept beside it.
"""

from __future__ import annotations

import math
from typing import Any, Iterable, Sequence

from .scouting_interpretation import assert_language

VERSION = "10.1"

# ============================================================================== sprint phases
# (phase_id, label, features, speed range in words). Each 09B curve feature belongs to exactly one phase.
SPRINT_PHASES: tuple[tuple[str, str, tuple[str, ...], str], ...] = (
    ("initial", "Initial acceleration", ("acc_2_4",), "first steps, tracked speed 2 → 4 yd/s"),
    ("early_dev", "Early speed development", ("acc_4_6", "speed_10yd"), "4 → 6 yd/s and the speed carried at 10 yd"),
    ("intermediate", "Intermediate speed development", ("time_4_8", "acc_6_8", "speed_20yd"), "4 → 8 yd/s build-up, 6 → 8 yd/s and the speed at 20 yd"),
    ("late", "Late sprint", ("speed_30yd",), "speed carried at 30 yd"),
    ("max", "Maximum tracked speed", ("top_speed",), "95th percentile of tracked speed inside the recording"),
)
PHASE_OF = {f: pid for pid, _, feats, _ in SPRINT_PHASES for f in feats}
UNUSABLE = ("unstable", "insufficient_support")

NO_DATA = {
    "no_tracked_attempt": "No tracked forty attempt is recorded for this prospect. Official testing only — nothing is imputed.",
    "missing_split": "Official 10-yard split not recorded. Comparisons that need it are not shown.",
    "single_attempt": "One recorded attempt: repeat-attempt stability cannot be estimated for this prospect.",
    "unsupported_demand": "This NFL movement demand has no physical analogue in the forty-yard dash. The Lab makes no statement about it.",
    "low_confidence": "Recording evidence is below the high tier: the measurement is shown, with that caveat.",
    "no_twin": "No stopwatch twin under the stored matching rule (same position, official forty within 0.03 s) and the current filters.",
    "no_nfl_observation": "Too few observed NFL episodes for this position and demand to describe it (support rule of the 09A atlas).",
    "no_feature": "This feature was not recorded for both athletes: no difference is shown.",
    "phase_not_identified": "Phase not identified in the recorded attempts of both athletes.",
}


def _num(x: Any) -> float | None:
    if x is None:
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(v) else v


def usable(d: dict[str, Any]) -> bool:
    """A difference row whose feature is repeatable enough to describe an athlete (09B / 08B.4 status)."""
    return d.get("evidence_status") not in (*UNUSABLE, None)


# ============================================================================== phase decomposition
def phase_decomposition(diffs: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """One row per sprint phase from the stored 09C differences (B − A).

    status: ``distinguishable`` (a usable feature of the phase exceeds its uncertainty) · ``not_distinguishable`` (tested,
    inside the uncertainty) · ``no_error_estimate`` (both values recorded, no SE) · ``not_identified`` (no feature of the
    phase recorded for both athletes). A phase is never filled from another phase."""
    by = {d["feature"]: d for d in diffs}
    out = []
    for pid, label, feats, words in SPRINT_PHASES:
        rows = []
        for f in feats:
            d = by.get(f)
            if d is None or _num(d.get("value_a")) is None or _num(d.get("value_b")) is None:
                continue
            rows.append({"feature": f, "label": d.get("label", f), "unit": d.get("unit", ""), "value_a": _num(d["value_a"]), "value_b": _num(d["value_b"]),
                         "difference": _num(d.get("difference")), "ci_lower": _num(d.get("ci_lower")), "ci_upper": _num(d.get("ci_upper")),
                         "z": _num(d.get("z")), "verdict": d.get("verdict"), "evidence_status": d.get("evidence_status"), "phase_09": d.get("phase"),
                         "quicker": d.get("quicker", ""), "usable": usable(d)})
        if not rows:
            status = "not_identified"
        elif any(r["verdict"] == "distinguishable" and r["usable"] for r in rows):
            status = "distinguishable"
        elif any(r["z"] is not None for r in rows):
            status = "not_distinguishable"
        else:
            status = "no_error_estimate"
        pool = [r for r in rows if r["verdict"] == "distinguishable" and r["usable"]] if status == "distinguishable" else [r for r in rows if r["z"] is not None]
        lead = max(pool, key=lambda r: abs(r["z"]), default=None)
        out.append({"phase_id": pid, "phase": label, "range": words, "status": status, "features": rows, "lead": lead,
                    "n_features": len(rows), "n_distinguishable": sum(r["verdict"] == "distinguishable" and r["usable"] for r in rows)})
    return out


def phase_sentence(p: dict[str, Any], names: tuple[str, str]) -> str:
    """One plain sentence per phase (the wording guard applies)."""
    if p["status"] == "not_identified":
        s = f"{p['phase']}: {NO_DATA['phase_not_identified']}"
    elif p["status"] == "no_error_estimate":
        s = f"{p['phase']}: recorded for both athletes, but without a repeat-attempt error estimate — no verdict."
    else:
        r = p["lead"]
        iv = "" if r["ci_lower"] is None else f" [{r['ci_lower']:+.2f}, {r['ci_upper']:+.2f}]"
        verdict = "larger than its measurement uncertainty" if p["status"] == "distinguishable" else "inside its measurement uncertainty — not a difference"
        s = (f"{p['phase']}: {r['label']} {r['value_a']:.2f} ({names[0]}) vs {r['value_b']:.2f} ({names[1]}) {r['unit']}, "
             f"difference {r['difference']:+.2f}{iv} — {verdict}.")
    assert_language(s)
    return s


# ============================================================================== evidence labels
def evidence_label(row: dict[str, Any] | None) -> dict[str, str]:
    """The five evidence answers of one feature (``feature_evidence`` row of the mart). Missing row → every answer says so."""
    if not row:
        return {"repeatability": "not assessed", "redundancy": "not assessed", "nfl_transfer": "not established", "coverage": "not assessed",
                "descriptive": "descriptive only", "short": "evidence not assessed", "tooltip": "No evidence row for this feature in the mart."}
    icc, r2 = _num(row.get("icc_within_position")), _num(row.get("r2_official_times"))
    rep = f"repeat-attempt ICC {icc:.2f}" if icc is not None else "repeat-attempt ICC not estimable"
    rep += f" ({row.get('repeatability_label') or row.get('evidence_status') or 'n/a'})"
    red = f"R² on official times {r2:.2f}" if r2 is not None else "R² on official times not estimable"
    red += f" ({row.get('redundancy_label') or 'n/a'})"
    cov = f"{row.get('n_athletes') or 0} athletes, {row.get('n_repeat') or 0} with a repeat attempt"
    status = str(row.get("evidence_status") or "n/a")
    short = {"repeatable_beyond_official": "repeatable · adds to official tests", "unstable": "not repeatable", "insufficient_support": "insufficient evidence"}.get(
        status, status.replace("_", " "))
    out = {"repeatability": rep, "redundancy": red, "nfl_transfer": "not established (Milestones 06 and 08B)", "coverage": cov,
           "descriptive": "descriptive measurement — football value " + str(row.get("football_value") or "not established"), "short": short}
    out["tooltip"] = " · ".join(f"{k}: {out[k]}" for k in ("repeatability", "redundancy", "nfl_transfer", "coverage", "descriptive")) + \
        f" · source: {row.get('evidence_source') or 'n/a'}"
    return out


# ============================================================================== repeat-attempt stability
def attempt_stability(metrics: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Per feature of one athlete (``prospect_metrics`` rows): the recorded range across attempts and its reading."""
    out = []
    for m in metrics:
        n = int(m.get("n_attempts") or 0)
        lo, hi, se = _num(m.get("attempt_min")), _num(m.get("attempt_max")), _num(m.get("standard_error"))
        if _num(m.get("value")) is None:
            reading = "not recorded"
        elif n < 2 or lo is None or hi is None:
            reading = "single attempt — stability not estimable"
        else:
            reading = f"range {lo:.2f}–{hi:.2f} {m.get('unit', '')} over {n} attempts".strip()
        out.append({"feature": m.get("feature"), "label": m.get("label"), "unit": m.get("unit"), "value": _num(m.get("value")), "n_attempts": n,
                    "attempt_min": lo, "attempt_max": hi, "spread": None if lo is None or hi is None else hi - lo, "standard_error": se, "reading": reading})
    return out


# ============================================================================== replay synchronisation
def _interp(xs: Sequence[float], ys: Sequence[float], x: float) -> float | None:
    """Linear interpolation inside [xs[0], xs[-1]]; None outside (no extrapolation). ``xs`` must increase."""
    if not xs or x < xs[0] - 1e-9 or x > xs[-1] + 1e-9:            # tolerance: floating-point edge of a grid point, not extrapolation
        return None
    x = min(max(x, xs[0]), xs[-1])
    lo, hi = 0, len(xs) - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if xs[mid] <= x:
            lo = mid
        else:
            hi = mid
    if xs[hi] == xs[lo]:
        return ys[lo]
    w = (x - xs[lo]) / (xs[hi] - xs[lo])
    return ys[lo] + w * (ys[hi] - ys[lo])


def sync_series(attempt: dict[str, Any], mode: str, step: float, metric: str) -> dict[str, Any]:
    """Place one recorded attempt (``{t, dist, speed, accel}`` lists) on a regular grid of elapsed time (``mode='time'``,
    seconds from the first recorded frame) or distance (``'distance'``, yards from the first recorded frame).

    Returns ``grid``, ``x_obs`` / ``y_obs`` (the recorded frames, untouched), ``progress`` (distance in time mode, elapsed
    time in distance mode) and ``value`` (the metric) on the grid, None beyond the recording, and ``interpolated``: True
    where a grid point falls between two recorded frames (the application discloses it)."""
    t, d = attempt.get("t") or [], attempt.get("dist") or []
    y = attempt.get({"speed": "speed", "accel": "accel"}.get(metric, "speed")) or []
    rows = [(float(a), float(b), float(c)) for a, b, c in zip(t, d, y) if _num(a) is not None and _num(b) is not None and _num(c) is not None]
    if len(rows) < 2:
        return {"grid": [], "x_obs": [], "y_obs": [], "progress": [], "value": [], "interpolated": [], "n_frames": len(rows)}
    t0, d0 = rows[0][0], rows[0][1]
    ts, ds, vs = [r[0] - t0 for r in rows], [r[1] - d0 for r in rows], [r[2] for r in rows]
    if mode == "distance":                                        # distance must increase for interpolation: keep the running maximum
        keep, best = [], -1.0
        for k, x in enumerate(ds):
            if x > best:
                keep.append(k)
                best = x
        ts, ds, vs = [ts[k] for k in keep], [ds[k] for k in keep], [vs[k] for k in keep]
        axis, other = ds, ts
    else:
        axis, other = ts, ds
    n = int(math.floor(axis[-1] / step + 1e-9)) + 1
    grid = [round(k * step, 6) for k in range(n)]
    obs = set(round(a, 6) for a in axis)
    return {"grid": grid, "x_obs": axis, "y_obs": vs, "progress": [_interp(axis, other, g) for g in grid], "value": [_interp(axis, vs, g) for g in grid],
            "interpolated": [round(g, 6) not in obs for g in grid], "n_frames": len(axis)}


# ============================================================================== text helpers
def words(text: str) -> int:
    return len([w for w in text.replace("—", " ").split() if any(ch.isalnum() for ch in w)])


def fit_words(sentences: Iterable[str], limit: int) -> str:
    """Join whole sentences while the total stays within ``limit`` words (the first sentence is always kept)."""
    out: list[str] = []
    for s in sentences:
        if not s:
            continue
        if out and words(" ".join([*out, s])) > limit:
            break
        out.append(s)
    text = " ".join(out)
    assert_language(text)
    return text


__all__ = ["NO_DATA", "PHASE_OF", "SPRINT_PHASES", "UNUSABLE", "VERSION", "attempt_stability", "evidence_label", "fit_words", "phase_decomposition",
           "phase_sentence", "sync_series", "usable", "words"]
