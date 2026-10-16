"""Milestone 10C — the NFL Movement Gap: what NFL games ask of each position, and what the forty-yard dash can describe.

Standard library only (mart rows as dicts), shared by notebook 10C and the application.

    COVERAGE_CLASSES        directly measured · partially comparable · not directly measured · unavailable (data)
    coverage_class          one demand × position, derived from the 09A registry analogue, the 09A support flag and the
                            09B matching summary — never a score between 0 and 1
    gap_matrix              every position × demand cell with its class, rate, counts and the reason for the class
    gap_summary             per position: exposure rate by class (episodes per 100 eligible player-plays)
    compare_positions       two positions on one demand, with the flags that explain a visually large difference
    TEST_PROPOSALS          candidate measurement protocols for demands the forty does not describe
    executive_summary       the one-page summary text

Exposure rates are episodes per 100 ELIGIBLE player-plays (the 05A denominator). Categories overlap (a re-acceleration is
also an acceleration episode; a repeated-change sequence contains episodes of other categories), so rates of different
demands are not additive shares of a play. Demand frequency is observed exposure, not importance to performance.
"""

from __future__ import annotations

from typing import Any, Iterable, Sequence

from .scouting_interpretation import assert_language

VERSION = "10.2"
# 10.2 (Milestone 11.0 correction): the 09A registry stores a missing analogue as the TEXT "none"; 10.1 read that text as a
# shared dimension, so braking, re-acceleration, redirection, speed maintenance and repeated changes were classed
# "unavailable (data limitation)" instead of "not directly measured". 10.2 ignores placeholder text and, independently of
# the registry text, declares the demands the forty cannot contain (no braking, no turn, no re-acceleration, no holding
# phase inside the timed run). Outputs written by 10.1 are kept as history; the corrected definitions are versioned in
# ``m11.inventory.movement_gap_registry_v2``.
PLACEHOLDER_ANALOGUES = {"", "none", "—", "-", "n/a", "na", "null"}
FORTY_CANNOT_MEASURE = ("braking", "reaccel_after_brake", "redirection", "repeated_change", "speed_maintenance")
DIRECT, PARTIAL, NOT_MEASURED, UNAVAILABLE = "direct", "partial", "not_measured", "unavailable"
COVERAGE_CLASSES = {
    DIRECT: "Directly measured by forty tracking",
    PARTIAL: "Partially physically comparable",
    NOT_MEASURED: "Not directly measured by the forty",
    UNAVAILABLE: "Unavailable (data limitation)",
}
CLASS_ORDER = (DIRECT, PARTIAL, NOT_MEASURED, UNAVAILABLE)

# Start-state agreement between the forty and the NFL episode, for demands WITH a registry analogue. Declared from the
# comparability / limitation text of the 09A registry (movement_demands.demand_specs), never from a distribution:
#   accel_low        the forty starts from rest; the episode starts below 2 yd/s — same start state, same bands.
#   accel_mid        "a standing-start band transit is not a rolling-start acceleration" — same bands, different state.
#   transition_high  the forty crosses 6 → 8 yd/s while already moving and accelerating — the episode's state.
#   sustained_high   the forty can end before top speed and records no holding phase — top speed is comparable only in part.
START_STATE_MATCH = {"accel_low": True, "accel_mid": False, "transition_high": True, "sustained_high": False}
MIN_ATHLETES_COMPARE = 30           # proposed: below it a position's rate is flagged as thinly supported in a comparison
EXPOSURE_RATIO_FLAG = 2.0           # proposed: eligible-play counts differing by more than this factor are flagged

TEST_PROPOSALS: dict[str, dict[str, str]] = {
    "braking": {"test": "Deceleration test", "protocol": "sprint 10–15 yd to a marked line at maximal effort, stop inside a 2-yd zone; at least two trials; "
                "tracked at ≥ 10 Hz; report peak and mean deceleration and stopping distance"},
    "reaccel_after_brake": {"test": "Acceleration after controlled braking", "protocol": "10-yd sprint, decelerate to near standstill inside a marked zone, "
                            "re-accelerate 10 yd; report the 2 → 4 and 4 → 6 yd/s band accelerations of the second acceleration with the same first-crossing rule"},
    "redirection": {"test": "Reactive change-of-direction test", "protocol": "cued 45° and 90° cuts at a controlled entry speed (light or partner cue); tracked heading "
                    "change, entry and exit speed; both directions, at least two trials each"},
    "repeated_change": {"test": "Repeated acceleration test", "protocol": "6 × (accelerate 10 yd, brake, turn) with fixed short rest; report the change of band "
                        "acceleration across repetitions"},
    "speed_maintenance": {"test": "Speed-holding segment", "protocol": "20-yd build-up followed by a 20-yd zone held at a steady speed; report speed variation inside the zone"},
    "sustained_high": {"test": "Flying sprint", "protocol": "a flying 20-yd segment after a 20–30-yd build-up, so top speed is reached and held inside the recording"},
    "accel_mid": {"test": "Rolling-start acceleration", "protocol": "enter a 10-yd zone at a controlled 4 yd/s and accelerate maximally; measures the 4 → 8 yd/s band from a rolling start"},
}
POSITION_SEQUENCE = {
    "WR": "release → stem at intermediate speed → brake → break at 45–90° → re-accelerate",
    "TE": "release from an attached stance → seam stem → brake → redirect → re-accelerate",
    "DB": "backpedal → plant (brake) → drive at 45° → close over 5–15 yd",
    "DL": "get-off from stance → 2-yd burst → redirect toward the quarterback → close",
    "OL": "pull or set from stance → short burst over 3–5 yd → redirect to the second level",
}
LIMIT = ("A proposed test is a candidate way to measure a demand the forty does not describe. Its value for evaluating NFL prospects has not been "
         "established and would need prospective evidence.")


def _analogue(reg: dict[str, Any]) -> list[str]:
    if str(reg.get("demand_id")) in FORTY_CANNOT_MEASURE:
        return []
    return [s.strip() for s in str(reg.get("combine_analogue") or "").split(",") if s.strip().lower() not in PLACEHOLDER_ANALOGUES]


def coverage_class(reg: dict[str, Any], cell: dict[str, Any] | None, n_matched: int) -> tuple[str, str]:
    """(class, reason) of one demand for one position.

    ``reg``: ``nfl_demand_registry`` row · ``cell``: ``nfl_movement_demands`` row of the position (None = absent) ·
    ``n_matched``: prospects of the position in ``movement_matching_summary`` for the demand (0 = no supported comparison)."""
    dims = _analogue(reg)
    if not dims:
        return NOT_MEASURED, f"no physical analogue in the forty — {reg.get('comparability') or 'the forty contains no such phase'}"
    if cell is None or not cell.get("supported"):
        n = 0 if cell is None else int(cell.get("n_episodes") or 0)
        return UNAVAILABLE, f"shared dimension(s) {', '.join(dims)} exist, but only {n:,} observed episode(s) for this position — below the 09A support rule"
    if n_matched <= 0:
        return UNAVAILABLE, f"shared dimension(s) {', '.join(dims)} exist, but no prospect of this position has a supported comparison (09B)"
    if START_STATE_MATCH.get(str(reg.get("demand_id")), False):
        return DIRECT, f"same speed band(s) {', '.join(dims)} and the same start state — {reg.get('comparability') or ''}".rstrip(" —")
    return PARTIAL, f"same dimension(s) {', '.join(dims)}, different movement state — {reg.get('comparability') or ''}".rstrip(" —")


def gap_matrix(registry: Sequence[dict[str, Any]], demands: Sequence[dict[str, Any]], summary: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Every (position, demand) cell. ``demands``: ``nfl_movement_demands`` rows (the ``ALL`` row is skipped)."""
    cells = {(r["position_group"], r["demand_id"]): r for r in demands}
    matched: dict[tuple[str, str], int] = {}
    for r in summary:
        k = (r["position_group"], r["demand_id"])
        matched[k] = max(matched.get(k, 0), int(r.get("n_prospects") or 0))
    positions = sorted({r["position_group"] for r in demands if r["position_group"] != "ALL"})
    out = []
    for p in positions:
        for reg in registry:
            d = reg["demand_id"]
            cell = cells.get((p, d))
            cls, why = coverage_class(reg, cell, matched.get((p, d), 0))
            out.append({"position_group": p, "demand_id": d, "label": reg.get("label", d), "coverage_class": cls, "class_label": COVERAGE_CLASSES[cls],
                        "reason": why, "combine_analogue": ", ".join(_analogue(reg)), "supported": bool(cell and cell.get("supported")),
                        "n_episodes": int((cell or {}).get("n_episodes") or 0), "n_athletes": int((cell or {}).get("n_athletes") or 0),
                        "n_eligible_plays": int((cell or {}).get("n_eligible_plays") or 0),
                        "episodes_per_100_plays": (cell or {}).get("episodes_per_100_plays"), "share_plays_with": (cell or {}).get("share_plays_with"),
                        "median_duration_s": (cell or {}).get("median_duration_s"), "median_distance_yd": (cell or {}).get("median_distance_yd"),
                        "n_matched_prospects": matched.get((p, d), 0), "test_proposal": TEST_PROPOSALS.get(d, {}).get("test", "")})
    return out


def gap_summary(matrix: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Per position: summed exposure rate by class (supported cells only) and the share of that rate the forty does not describe.

    Rates of overlapping categories are summed, so ``rate_total`` is an exposure INDEX, not episodes of distinct movement;
    the share compares like with like (same position, same denominator)."""
    out = []
    for p in sorted({r["position_group"] for r in matrix}):
        rows = [r for r in matrix if r["position_group"] == p]
        rate = {c: sum(float(r["episodes_per_100_plays"] or 0.0) for r in rows if r["coverage_class"] == c and r["supported"]) for c in CLASS_ORDER}
        total = sum(rate.values())
        out.append({"position_group": p, **{f"rate_{c}": rate[c] for c in CLASS_ORDER}, "rate_total": total,
                    "share_not_measured": None if total <= 0 else rate[NOT_MEASURED] / total,
                    "share_direct_or_partial": None if total <= 0 else (rate[DIRECT] + rate[PARTIAL]) / total,
                    "n_supported_demands": sum(r["supported"] for r in rows), "n_eligible_plays": max((r["n_eligible_plays"] for r in rows), default=0),
                    "min_athletes": min((r["n_athletes"] for r in rows if r["supported"]), default=0),
                    "n_not_measured": sum(r["coverage_class"] == NOT_MEASURED for r in rows), "n_unavailable": sum(r["coverage_class"] == UNAVAILABLE for r in rows)})
    return out


def gap_position(summary: Sequence[dict[str, Any]], min_athletes: int = MIN_ATHLETES_COMPARE) -> dict[str, Any] | None:
    """The position whose observed exposure is least described by the forty (largest ``share_not_measured``), among
    positions whose supported cells all have ≥ ``min_athletes`` athletes. Rule declared before reading the mart."""
    ok = [s for s in summary if s["share_not_measured"] is not None and s["min_athletes"] >= min_athletes]
    return max(ok, key=lambda s: (s["share_not_measured"], s["position_group"]), default=None)


def compare_positions(matrix: Sequence[dict[str, Any]], a: str, b: str, demand: str) -> dict[str, Any]:
    """Two positions on one demand: rates, counts, rate ratio, and the flags behind a visually large difference."""
    get = lambda p: next((r for r in matrix if r["position_group"] == p and r["demand_id"] == demand), None)
    ra, rb = get(a), get(b)
    flags = []
    for p, r in ((a, ra), (b, rb)):
        if r is None or not r["supported"]:
            flags.append(f"{p}: too few observed episodes to describe this demand")
        elif r["n_athletes"] < MIN_ATHLETES_COMPARE:
            flags.append(f"{p}: only {r['n_athletes']} athletes contribute — the rate rests on few players")
    if ra and rb and ra["n_eligible_plays"] and rb["n_eligible_plays"]:
        q = max(ra["n_eligible_plays"], rb["n_eligible_plays"]) / max(1, min(ra["n_eligible_plays"], rb["n_eligible_plays"]))
        if q > EXPOSURE_RATIO_FLAG:
            flags.append(f"eligible player-plays differ {q:.1f}-fold ({ra['n_eligible_plays']:,} vs {rb['n_eligible_plays']:,}): raw episode counts are not comparable — read the rate")
    ratio = None
    if ra and rb and ra["supported"] and rb["supported"] and ra["episodes_per_100_plays"] and rb["episodes_per_100_plays"]:
        ratio = float(rb["episodes_per_100_plays"]) / float(ra["episodes_per_100_plays"])
    flags.append("episodes within a player are dependent: athletes, not episodes, are the sample size")
    for f in flags:
        assert_language(f)
    return {"a": ra, "b": rb, "rate_ratio_b_over_a": ratio, "flags": flags}


def proposals(matrix: Sequence[dict[str, Any]], position: str) -> list[dict[str, str]]:
    """Candidate tests for the supported demands of a position that the forty does not describe (or describes in part)."""
    rows = sorted((r for r in matrix if r["position_group"] == position and r["supported"] and r["coverage_class"] in (NOT_MEASURED, PARTIAL)),
                  key=lambda r: -float(r["episodes_per_100_plays"] or 0.0))
    out = [{"demand_id": r["demand_id"], "demand": r["label"], "rate": f"{float(r['episodes_per_100_plays'] or 0):.1f} per 100 eligible player-plays",
            "class": r["class_label"], **TEST_PROPOSALS[r["demand_id"]]} for r in rows if r["demand_id"] in TEST_PROPOSALS]
    if position in POSITION_SEQUENCE:
        out.append({"demand_id": "sequence", "demand": "position-specific movement sequence", "rate": "combines the demands above", "class": "proposed",
                    "test": f"{position} movement sequence", "protocol": POSITION_SEQUENCE[position] + "; tracked at ≥ 10 Hz with the same band and braking rules as the NFL atlas"})
    return out


def executive_summary(summary: Sequence[dict[str, Any]], matrix: Sequence[dict[str, Any]], focus: dict[str, Any] | None, n_episodes: int) -> list[str]:
    """One-page summary lines (numbers from the mart rows passed in)."""
    labels = {r["demand_id"]: r["label"] for r in matrix}
    with_analogue = sorted({r["demand_id"] for r in matrix if r["combine_analogue"]})
    without = [labels[d] for d in labels if d not in with_analogue]
    lines = [f"The forty-yard dash is one straight-line acceleration from rest. Across {n_episodes:,} observed NFL movement episodes, the 09A atlas defines "
             f"{len(labels)} speed-change demands; {len(with_analogue)} have a physical analogue in forty tracking and {len(without)} have none "
             f"({', '.join(without)})."]
    for s in summary:
        if s["share_not_measured"] is None:
            continue
        lines.append(f"{s['position_group']}: {100 * s['share_not_measured']:.0f} % of the summed exposure rate falls in demands the forty does not describe "
                     f"({s['rate_not_measured']:.1f} of {s['rate_total']:.1f} episodes per 100 eligible player-plays; {s['n_eligible_plays']:,} eligible player-plays).")
    if focus:
        lines.append(f"Largest gap: {focus['position_group']} — the position whose observed game movement the forty describes least, by this exposure index.")
    lines += ["Categories overlap, so the exposure index is not a share of distinct movements; frequency is exposure, not importance to performance.", LIMIT]
    for s in lines:
        assert_language(s)
    return lines


__all__ = ["CLASS_ORDER", "COVERAGE_CLASSES", "DIRECT", "EXPOSURE_RATIO_FLAG", "FORTY_CANNOT_MEASURE", "LIMIT", "PLACEHOLDER_ANALOGUES", "MIN_ATHLETES_COMPARE", "NOT_MEASURED", "PARTIAL", "POSITION_SEQUENCE",
           "START_STATE_MATCH", "TEST_PROPOSALS", "UNAVAILABLE", "VERSION", "compare_positions", "coverage_class", "executive_summary", "gap_matrix",
           "gap_position", "gap_summary", "proposals"]
