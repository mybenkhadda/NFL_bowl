"""Phase 1 — formal specifications of the M14 cross-sport movement metrics.

Every metric is declared BEFORE any data is seen: origin sport and concept, formula, required channels, unit, level
(event / sequence / athlete / train-fitted athlete), sampling assumptions, event definition, failure cases, football
positions where it is expected to matter, the reliability requirement it must pass, the NFL prediction hypothesis and
the sign convention used by composites (+1 = higher is the hypothesised better). Nothing in this module reads data.

Channels available in the M12 bundle (``m11.representation.CHANNELS``): speed (yd/s), a_tan (yd/s², signed),
a_n (yd/s², signed centripetal), yaw (rad/s heading rate), dt (s), heading_valid. There are NO x / y positions, NO body
orientation and NO synchronised opponents / ball: metrics that need them are declared ``blocked`` and never computed
(nothing is fabricated).

Interpretive limits that apply to every metric (repeated in the reports):
* kinematic proxies only — no muscle force, power, tendon stiffness or neural drive is inferred;
* no reaction time — there is no stimulus timestamp;
* asymmetry is a movement description, never an injury-risk indicator;
* smoothness (SPARC, jerk) is descriptive, never a causal mechanism;
* 10 Hz sampling: phase timings are resolved to ~0.1 s (linear interpolation where stated).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable, Sequence

CHANNELS = ("speed", "a_tan", "a_n", "yaw", "dt", "heading_valid")
SCOPES: dict[str, tuple[str, ...]] = {"forty": ("forty",), "agility": ("three_cone", "short_shuttle", "long_shuttle"),
                                      "drill": ("pass_rush", "gauntlet", "backpedal_transition")}
NFL_CUT_KINDS = ("braking", "reaccel_after_brake", "redirection")
LEVELS = ("event", "sequence", "athlete", "athlete_trainfit", "blocked")
ALL_POSITIONS = ("ALL",)


@dataclass(frozen=True)
class MetricSpec:
    name: str
    family: str                  # motorsport · alpine · court · rugby_soccer · sprint · nfl_context
    origin_sport: str
    origin_concept: str
    formula: str
    channels: tuple[str, ...]
    unit: str
    level: str
    sampling: str
    event_definition: str
    failure_cases: str
    positions: tuple[str, ...]
    reliability_requirement: str
    hypothesis: str
    sign: int                    # +1 higher = hypothesised better, −1 lower = better, 0 = no direction (descriptive)
    scopes: tuple[str, ...] = ()  # Combine scopes where a predictor column is built
    braking_core: bool = False   # member of the pre-declared Hidden-Braking-Advantage signature
    requires: tuple[str, ...] = ()  # non-channel inputs (athlete columns, synchronised data)
    blocked_reason: str = ""
    caveat: str = ""


_RELIAB = "ICC(1,1) of per-sequence values ≥ threshold (CI lower ≥ marginal), ≥ min athletes with ≥ 2 trials, perturbation Spearman ≥ 0.8"
_EV = ("cut event: valley of speed inside one gap-free valid segment; entry peak ≥ entry_min_speed; drop ≥ max(drop_min, drop_frac·v_entry); "
       "closed by a rise ≥ rise_min (else exit-censored)")
_CUT_POS = ("DB", "WR", "LB", "RB", "S", "CB")
_SPRINT_POS = ("ALL",)

METRICS: tuple[MetricSpec, ...] = (
    # ------------------------------------------------------------------ motorsport
    MetricSpec("corner_exit_efficiency", "motorsport", "motorsport", "corner exit speed decides the next straight",
               "v(t_min + 0.5 s) / v_entry", ("speed", "dt"), "ratio", "event", "≥ 5 valid frames after the minimum (0.5 s at 10 Hz)", _EV,
               "segment ends < 0.5 s after the minimum; v_entry < entry_min_speed", _CUT_POS, _RELIAB,
               "athletes who carry more speed out of a cut at the Combine show higher NFL movement capacity at equal M1", +1, ("agility", "drill"),
               braking_core=True),
    MetricSpec("braking_point_efficiency", "motorsport", "motorsport", "late braking: shed the speed in the shortest distance",
               "(v_b² − v_min²) / (2 · d_brake), d_brake = ∫_{t_b}^{t_min} v dt (trapezoid)", ("speed", "a_tan", "dt"), "yd/s²", "event",
               "braking onset resolved to one frame; distance by trapezoid on valid frames", _EV + "; braking onset = first sustained a_tan ≤ −brake_thresh after the entry peak",
               "no sustained braking (weak_braking); d_brake < 0.05 yd", _CUT_POS, _RELIAB,
               "equivalent constant deceleration over the braking distance adds beyond forty / shuttle times", +1, ("agility", "drill"),
               braking_core=True, caveat="work-energy equivalent deceleration — NOT a force; mass is not used"),
    MetricSpec("braking_budget_utilization", "motorsport", "motorsport", "square brake trace: how much of the braking phase is spent near peak",
               "mean(−a_tan) / max(−a_tan) over the braking phase [t_b, t_min]", ("a_tan",), "ratio (0–1]", "event",
               "≥ 2 braking frames", _EV, "peak braking < brake_thresh; single braking frame", _CUT_POS, _RELIAB,
               "a plateau-shaped deceleration (high utilisation) relates to NFL braking capacity", +1, ("agility", "drill"), braking_core=True),
    MetricSpec("stop_go_elasticity", "motorsport", "motorsport / court", "stop-go: how much of the braking intensity returns as propulsion",
               "mean a_tan over (t_min, t_min + 0.5 s] / mean(−a_tan) over [t_min − 0.5 s, t_min]", ("a_tan", "dt"), "ratio", "event",
               "0.5 s windows (≈ 5 frames) on both sides of the minimum", _EV, "mean braking in the window < 0.5 yd/s²; window truncated", _CUT_POS,
               _RELIAB, "athletes who convert braking into immediate propulsion transfer better", +1, ("agility", "drill"),
               caveat="named 'elasticity PROXY': no tendon / stretch-shortening inference"),
    # ------------------------------------------------------------------ alpine skiing
    MetricSpec("turn_speed_retention", "alpine", "alpine skiing", "carry speed through the turn",
               "v_min / v_entry", ("speed",), "ratio", "event", "minimum at 10 Hz may miss the true minimum by < 0.1 s", _EV,
               "v_entry < entry_min_speed", _CUT_POS, _RELIAB, "higher retention at equal times = hidden cutting capacity", +1, ("agility", "drill"),
               braking_core=True),
    MetricSpec("curvature_speed_envelope", "alpine", "alpine skiing", "lateral (centripetal) capacity across speeds",
               "mean over speed bins [2,4) [4,6) [6,8) [8,∞) yd/s of q90 |a_n| (bins with ≥ 3 frames, ≥ 2 bins)", ("speed", "a_n"), "yd/s²",
               "sequence", "a_n from heading change × speed (provider); v ≥ 2 yd/s avoids the 1/v² blow-up", "whole sequence (no event needed)",
               "< 2 populated speed bins", _CUT_POS, _RELIAB, "a higher centripetal envelope at the Combine relates to NFL redirection capacity", +1,
               ("agility", "drill")),
    MetricSpec("multi_cut_consistency", "alpine", "alpine skiing (slalom)", "repeatable turns gate after gate",
               "SD of corner_exit_efficiency across the events of one sequence (≥ 2 events)", ("speed", "dt"), "ratio SD", "sequence",
               "as corner_exit_efficiency", _EV, "< 2 events with an exit value in the sequence", _CUT_POS, _RELIAB,
               "lower variability across cuts relates to NFL movement capacity", -1, ("agility",)),
    MetricSpec("directional_asymmetry", "alpine", "alpine skiing", "left vs right turn quality",
               "|median CEE(+) − median CEE(−)| / mean(two medians); direction = sign of ∫ yaw dt over the event (|angle| ≥ 20°)",
               ("speed", "yaw", "heading_valid", "dt"), "ratio", "athlete", "turn direction needs heading-valid coverage ≥ 0.5 of the event",
               _EV, "< 2 events in either direction; heading invalid near standstill", _CUT_POS,
               "split-half (odd vs even sequences) Spearman–Brown ≥ threshold", "descriptive only (no direction)", 0, ("agility",),
               caveat="magnitude only (sign convention of yaw not verified across domains); NEVER an injury-risk indicator"),
    # ------------------------------------------------------------------ tennis / basketball
    MetricSpec("brake_propulsion_transition", "court", "tennis / basketball", "split-step: time from braking release to propulsion",
               "t(a_tan ↑ +brake_thresh) − t(a_tan ↑ −brake_thresh), both crossings linearly interpolated after the last braking frame",
               ("a_tan", "dt"), "s", "event", "sub-frame by linear interpolation; resolution still limited by 10 Hz smoothing", _EV,
               "no propulsion onset (propulsion_censored); weak braking", _CUT_POS, _RELIAB,
               "shorter transitions at equal times relate to NFL movement capacity", -1, ("agility", "drill"), braking_core=True,
               caveat="movement transition only — NOT a reaction time (no stimulus timestamp)"),
    MetricSpec("first_step_explosiveness", "court", "tennis / basketball", "first step out of the plant",
               "mean a_tan over the first 0.3 s from propulsion onset", ("a_tan", "dt"), "yd/s²", "event", "3 frames at 10 Hz", _EV,
               "propulsion_censored; < 3 frames after onset", _CUT_POS, _RELIAB, "stronger first steps after a cut add beyond the ten split", +1,
               ("agility", "drill"), caveat="kinematic acceleration; no force / power claim"),
    MetricSpec("cut_recovery_time", "court", "tennis / basketball", "recovery to useful speed after the plant",
               "t(v ≥ recovery_frac · v_entry) − t_min (crossing interpolated)", ("speed", "dt"), "s", "event", "crossing interpolated", _EV,
               "speed never regains the threshold inside the segment (recovery_censored)", _CUT_POS, _RELIAB,
               "faster recovery after a cut relates to NFL movement capacity", -1, ("agility", "drill"), braking_core=True,
               caveat="stands in for 'lateral recovery efficiency', which is blocked"),
    MetricSpec("lateral_recovery_efficiency", "court", "tennis / basketball", "lateral shuffle recovery",
               "needs lateral displacement relative to body orientation", ("x", "y", "orientation"), "—", "blocked", "—", "—", "—", _CUT_POS, "—",
               "—", 0, (), requires=("x", "y", "orientation"), blocked_reason="no x / y positions and no body orientation in the bundle"),
    MetricSpec("sparc_smoothness", "court", "motor control (tennis / clinical)", "spectral arc length of the speed profile",
               "SPARC = −Σ √((Δf/f_c)² + (Δ|V̂(f)|)²) over f ≤ min(10 Hz, Nyquist) with the 0.05 amplitude threshold",
               ("speed", "dt"), "dimensionless (≤ 0; closer to 0 = smoother)", "sequence",
               "10 Hz → Nyquist 5 Hz; provider smoothing flattens the spectrum (compare like with like)", "longest gap-free segment ≥ 10 frames",
               "segment < 10 frames; flat profile", _SPRINT_POS, _RELIAB, "smoother sprint / agility profiles relate to NFL capacity", +1,
               ("forty", "agility"), caveat="descriptive smoothness — NEVER a causal mechanism"),
    # ------------------------------------------------------------------ rugby / soccer
    MetricSpec("cod_deficit_residual", "rugby_soccer", "rugby / soccer", "change-of-direction deficit",
               "official agility time − f(forty, weight), f = OLS fitted on TRAINING athletes of each outer fold", (), "s", "athlete_trainfit",
               "official times (hand / electronic as recorded)", "athlete level", "official agility time missing", _CUT_POS,
               "official-time reliability not estimable here (one recorded value)", "a lower deficit at equal straight-line speed relates to NFL capacity", -1,
               (), requires=("three_cone|short_shuttle", "forty", "weight")),
    MetricSpec("speed_adjusted_cutting", "rugby_soccer", "rugby / soccer", "cut quality adjusted for approach speed",
               "athlete median turn_speed_retention − g(athlete median v_entry), g = OLS on TRAINING athletes", ("speed",), "ratio residual",
               "athlete_trainfit", "as turn_speed_retention", _EV, "no agility events", _CUT_POS, "inherits turn_speed_retention",
               "retention beyond what entry speed explains adds beyond M1", +1, ("agility",)),
    MetricSpec("max_controlled_decel", "rugby_soccer", "rugby / soccer", "highest deceleration that still ends in a controlled change",
               "max over the braking phase of the 2-frame mean of −a_tan, completed events only", ("a_tan",), "yd/s²", "event",
               "2-frame mean suppresses single-frame spikes", _EV + "; event must be complete (braking, propulsion, exit observed)",
               "incomplete event", _CUT_POS, _RELIAB, "higher controlled deceleration capacity relates to NFL braking capacity", +1,
               ("agility", "drill"), braking_core=True, caveat="kinematic deceleration — no force inference"),
    MetricSpec("cut_recovery_asymmetry", "rugby_soccer", "rugby / soccer", "left vs right recovery after the cut",
               "|median recovery(+) − median recovery(−)| / mean(two medians)", ("speed", "yaw", "heading_valid", "dt"), "ratio", "athlete",
               "as directional_asymmetry", _EV, "< 2 events in either direction", _CUT_POS, "split-half (odd vs even sequences)",
               "descriptive only (no direction)", 0, ("agility",), caveat="NEVER an injury-risk indicator"),
    # ------------------------------------------------------------------ track sprinting
    MetricSpec("accel_envelope_a0", "sprint", "track sprinting", "acceleration–speed profile intercept (A0)",
               "a_tan = A0 + s·v fitted by least squares over acceleration-phase frames (start → peak speed, v ≥ 1 yd/s, a_tan > 0); A0 if s < 0",
               ("speed", "a_tan"), "yd/s²", "sequence", "≥ 5 frames in the acceleration phase", "forty start (any start speed)",
               "< 5 acceleration frames; slope ≥ 0 (profile undefined)", _SPRINT_POS, _RELIAB,
               "a higher theoretical acceleration intercept adds beyond forty / ten", +1, ("forty",),
               caveat="mass-free kinematic profile — no horizontal force / power claim"),
    MetricSpec("accel_envelope_v0", "sprint", "track sprinting", "acceleration–speed profile speed intercept (V0)",
               "V0 = −A0 / s (extrapolated speed at zero acceleration)", ("speed", "a_tan"), "yd/s", "sequence", "as accel_envelope_a0",
               "as accel_envelope_a0", "as accel_envelope_a0; extrapolation", _SPRINT_POS, _RELIAB, "a higher speed intercept adds beyond forty", +1,
               ("forty",), caveat="extrapolated — interpret with the fit R²"),
    MetricSpec("accel_reserve_at_vref", "sprint", "track sprinting", "speed-conditioned acceleration reserve",
               "A0 + s · v_ref (v_ref = 7 yd/s) when the phase reaches v_ref (no extrapolation beyond observed speed)", ("speed", "a_tan"), "yd/s²",
               "sequence", "as accel_envelope_a0", "as accel_envelope_a0", "peak speed < v_ref", _SPRINT_POS, _RELIAB,
               "acceleration still available at 7 yd/s adds beyond forty", +1, ("forty",)),
    MetricSpec("time_to_useful_speed", "sprint", "track sprinting", "time to a sport-useful absolute speed",
               "t(v ≥ useful_speed = 6 yd/s) − t_0 (crossing interpolated), start below 1 yd/s", ("speed", "dt"), "s", "sequence",
               "standing start required", "forty start", "start ≥ 1 yd/s; threshold never reached", _SPRINT_POS, _RELIAB,
               "reaching useful speed sooner adds beyond the ten split", -1, ("forty",)),
    MetricSpec("high_speed_accel", "sprint", "track sprinting", "acceleration capability near top speed",
               "mean a_tan over acceleration-phase frames with v ≥ 0.8 · peak speed (≥ 2 frames)", ("speed", "a_tan"), "yd/s²", "sequence",
               "as accel_envelope_a0", "forty", "< 2 qualifying frames", _SPRINT_POS, _RELIAB, "acceleration retained at high speed adds beyond forty", +1,
               ("forty",)),
    # ------------------------------------------------------------------ NFL contextual (multi-player) — blocked
    MetricSpec("pursuit_interception_efficiency", "nfl_context", "NFL tracking", "closing on the ball carrier",
               "time-to-intercept vs optimal pursuit angle", ("x", "y", "opponent_x", "opponent_y"), "—", "blocked", "—", "—", "—",
               ("LB", "S", "CB", "DL"), "—", "—", 0, (), requires=("x", "y", "synchronized opponents"),
               blocked_reason="no synchronised multi-player x / y tracking in the bundle"),
    MetricSpec("separation_creation_efficiency", "nfl_context", "NFL tracking", "separation gained per unit of movement",
               "Δ receiver–defender distance per break", ("x", "y", "opponent_x", "opponent_y"), "—", "blocked", "—", "—", "—", ("WR", "TE", "RB"),
               "—", "—", 0, (), requires=("x", "y", "synchronized opponents"), blocked_reason="no synchronised multi-player x / y tracking in the bundle"),
    MetricSpec("defensive_recovery_efficiency", "nfl_context", "NFL tracking", "recovering lost separation",
               "separation regained per second after the receiver break", ("x", "y", "opponent_x", "opponent_y", "ball"), "—", "blocked", "—", "—", "—",
               ("CB", "S"), "—", "—", 0, (), requires=("x", "y", "synchronized opponents", "ball"),
               blocked_reason="no synchronised multi-player x / y tracking or ball events in the bundle"),
)
BY_NAME = {m.name: m for m in METRICS}
FAMILIES = tuple(dict.fromkeys(m.family for m in METRICS))
BRAKING_CORE = tuple(m.name for m in METRICS if m.braking_core)
EVENT_METRICS = tuple(m.name for m in METRICS if m.level == "event")
SEQUENCE_METRICS = tuple(m.name for m in METRICS if m.level == "sequence")
ATHLETE_METRICS = tuple(m.name for m in METRICS if m.level == "athlete")
TRAINFIT_METRICS = tuple(m.name for m in METRICS if m.level == "athlete_trainfit")
BLOCKED_METRICS = tuple(m.name for m in METRICS if m.level == "blocked")


def spec(name: str) -> MetricSpec:
    return BY_NAME[name]


def combine_column(metric: str, scope: str) -> str:
    return f"m14c_{scope}__{metric}"


def nfl_column(metric: str) -> str:
    return f"m14n__{metric}"


def parse_column(col: str) -> tuple[str, str] | None:
    """``m14c_<scope>__<metric>`` → (scope, metric); anything else → None."""
    if not col.startswith("m14c_") or "__" not in col:
        return None
    scope, metric = col.removeprefix("m14c_").split("__", 1)
    return (scope, metric) if scope in SCOPES else None


def predictor_columns(metrics: Iterable[str] | None = None) -> list[str]:
    """Declared Combine predictor columns (event / sequence / athlete metrics × their scopes)."""
    names = list(metrics) if metrics is not None else [m.name for m in METRICS]
    return [combine_column(n, s) for n in names for s in BY_NAME[n].scopes if BY_NAME[n].level in ("event", "sequence", "athlete")]


def family_columns(columns: Sequence[str]) -> dict[str, list[str]]:
    """{family: present predictor columns} (train-fitted metrics are added by the transfer blocks, not here)."""
    out: dict[str, list[str]] = {}
    for c in columns:
        p = parse_column(c)
        if p and p[1] in BY_NAME:
            out.setdefault(BY_NAME[p[1]].family, []).append(c)
    return out


def definitions_frame() -> Any:
    import polars as pl

    rows = []
    for m in METRICS:
        d = asdict(m)
        for k in ("channels", "positions", "scopes", "requires"):
            d[k] = ", ".join(d[k])
        rows.append(d)
    return pl.DataFrame(rows, infer_schema_length=None)


def definitions_json() -> list[dict[str, Any]]:
    return [asdict(m) for m in METRICS]


def availability(channels: Sequence[str] = CHANNELS, athlete_columns: Sequence[str] = (), synchronized: bool = False) -> Any:
    """One row per metric: computable / blocked with the missing inputs. ``synchronized`` = multi-player tracking with
    positions exists (never true for the M12 bundle)."""
    import polars as pl

    have = set(channels)
    acols = set(athlete_columns)
    rows = []
    for m in METRICS:
        missing = [c for c in m.channels if c not in have]
        req_missing = []
        for r in m.requires:
            if "|" in r:
                if not any(x in acols for x in r.split("|")):
                    req_missing.append(r)
            elif r.startswith("synchronized") or r == "ball":
                if not synchronized:
                    req_missing.append(r)
            elif r in ("x", "y", "orientation"):
                if r not in have:
                    req_missing.append(r)
            elif r not in acols:
                req_missing.append(r)
        miss = sorted(set(missing) | set(req_missing))
        status = "blocked" if (m.level == "blocked" or miss) else "computable"
        reason = m.blocked_reason or (f"missing inputs: {', '.join(miss)}" if miss else "")
        rows.append({"metric": m.name, "family": m.family, "level": m.level, "status": status, "missing_inputs": ", ".join(miss), "reason": reason})
    return pl.DataFrame(rows, infer_schema_length=None)


__all__ = ["ATHLETE_METRICS", "BLOCKED_METRICS", "BRAKING_CORE", "BY_NAME", "CHANNELS", "EVENT_METRICS", "FAMILIES", "LEVELS", "METRICS", "MetricSpec",
           "NFL_CUT_KINDS", "SCOPES", "SEQUENCE_METRICS", "TRAINFIT_METRICS", "availability", "combine_column", "definitions_frame", "definitions_json",
           "family_columns", "nfl_column", "parse_column", "predictor_columns", "spec"]
