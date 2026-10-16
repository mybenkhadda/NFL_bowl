"""Milestone 09D — scouting interpretation engine: turns measured values into qualified scouting language.

Standard library only (plain dicts in, text out), so the notebooks AND the interactive application use the same code.

    INTERPRETATIONS        per metric: what is measured · physical meaning · difference from official testing ·
                           related football movements · what to look at next · what cannot be concluded
    interpretation_registry    the same, joined with the executed reliability / redundancy evidence
    pair_reading           Observation · Potential football relevance · Scouting follow-up · Limitations of one pair
    case_views             the scout, position-coach and general-manager views of one case
    prospect_report        the sections of a one-prospect scouting report
    render_html            a self-contained HTML report
    assert_language        the wording guard: no grade, no ranking label, no causal claim

Every sentence is assembled from measured values and their verdicts. Wording rule: an observed difference is stated as
observed; its football advantage is stated as NOT established. No rule of the form "quicker X means better player".
"""

from __future__ import annotations

import html
from typing import Any, Iterable, Sequence

VERSION = "1"
NOT_ESTABLISHED = "The corresponding football advantage has not been established."
DISCLAIMER = ("Descriptive scouting information from Combine tracking. No value on this page predicts NFL performance: in this project, tracked Combine "
              "features did not improve held-out prediction of NFL movement or receiver separation beyond official testing. Use it to decide what to "
              "examine next — not to rank prospects.")
FORBIDDEN_PHRASES = ("draft grade", "hidden star", "sleeper", "draft bust", "guarantee", "better receiver", "better player", "better prospect", "will succeed",
                     "nfl potential", "talent score", "translates to the nfl", "projects as")
PHASES = {
    "early": {"label": "early acceleration (first steps, below about 4 yd/s)", "demand": "accel_low",
              "relevance": "movements that start from rest or near rest: the first steps after the snap and short-area starts",
              "test": "timed starts over 0–5 and 0–10 yd with split timing, at least two repetitions, recorded from before the first movement"},
    "middle": {"label": "mid-sprint speed development (about 4 to 8 yd/s)", "demand": "accel_mid",
               "relevance": "building speed while already moving: movements that develop over several yards after the first steps",
               "test": "rolling-start (flying) 10-yard segments entered at a controlled speed, so the 4 → 8 yd/s band is measured directly"},
    "late": {"label": "late acceleration and high speed (above about 6 yd/s)", "demand": "transition_high",
             "relevance": "reaching and holding high speed over distance: long pursuits and movements down the field",
             "test": "a longer sprint or flying 20-yard segment, because 40 yards can end before top speed is reached"},
    "whole": {"label": "the whole run", "demand": "", "relevance": "no single movement phase", "test": "a repeat of the tracked forty"},
}
POSITION_FILM = {
    "WR": {"early": "releases at the line and the first three steps of short movements", "middle": "the stem of intermediate movements before the break",
           "late": "vertical movements and long runs after the catch"},
    "TE": {"early": "releases from an in-line or attached alignment", "middle": "movements up the seam through the second level",
           "late": "long runs after the catch and vertical movements"},
    "DB": {"early": "the first steps out of the backpedal or stance when the play declares", "middle": "closing from depth over 5 to 15 yards",
           "late": "carrying a vertical movement and long pursuit"},
    "DL": {"early": "the first two steps after the snap", "middle": "closing on the quarterback once free", "late": "backside pursuit over distance"},
    "OL": {"early": "the first steps of a pull or a set", "middle": "pulls and climbs to the second level", "late": "screens and long pursuit after a turnover"},
}
INTERPRETATIONS: dict[str, dict[str, str]] = {
    "cmb_forty": {"measured": "official time over 40 yards from a stationary start", "physical": "average speed over the whole run in one number",
                  "vs_official": "it IS the official test", "football": "general straight-line speed", "next": "use as today",
                  "cannot": "how the time was built: two different speed curves can give the same time"},
    "cmb_ten_yd_split": {"measured": "official time over the first 10 yards", "physical": "early acceleration as timed by the stopwatch",
                         "vs_official": "it IS an official split", "football": "short-area starts", "next": "use as today",
                         "cannot": "the shape of the acceleration inside those 10 yards"},
    "acc_2_4": {"measured": "mean acceleration while tracked speed rises from 2 to 4 yd/s", "physical": "how hard speed is built in the first steps",
                "vs_official": "inside the 10-yard split, which mixes this band with reaction and the next bands", "football": PHASES["early"]["relevance"],
                "next": "film of first steps; a timed start with split timing", "cannot": "whether it is used, or matters, in a game; the band spans very few 10 Hz frames"},
    "acc_4_6": {"measured": "mean acceleration while tracked speed rises from 4 to 6 yd/s", "physical": "how hard speed is built in the middle of the acceleration",
                "vs_official": "not isolated by any official split", "football": PHASES["middle"]["relevance"],
                "next": "film of movements that develop over several yards; a rolling-start test", "cannot": "performance from a rolling start: the forty passes this band once, from rest"},
    "acc_6_8": {"measured": "mean acceleration while tracked speed rises from 6 to 8 yd/s", "physical": "how hard speed is still built late in the acceleration",
                "vs_official": "not isolated by any official split", "football": PHASES["late"]["relevance"],
                "next": "film of long movements; a longer sprint", "cannot": "top-end speed beyond 40 yards"},
    "time_4_8": {"measured": "time for tracked speed to rise from 4 to 8 yd/s", "physical": "how quickly speed builds through the middle of the run",
                 "vs_official": "partly explained by the official times; repeatable after they are accounted for (Milestone 08B.4)",
                 "football": PHASES["middle"]["relevance"], "next": "compare film of mid-distance movements; a rolling-start test",
                 "cannot": "any NFL outcome: no incremental prediction was established (Milestones 06 and 08B)"},
    "top_speed": {"measured": "95th percentile of tracked speed in the forty", "physical": "the speed actually reached and held",
                  "vs_official": "largely the same information as the official forty", "football": "sustained high-speed running",
                  "next": "confirms or questions the official time; little new information", "cannot": "the athlete's true maximum: 40 yards may end before it"},
    "speed_10yd": {"measured": "tracked speed when 10 yards are covered", "physical": "speed already built after the first 10 yards",
                   "vs_official": "the 10-yard split gives the TIME to that point; this gives the SPEED at it", "football": PHASES["early"]["relevance"],
                   "next": "film of short-area movements", "cannot": "anything about speed later in the run"},
    "speed_20yd": {"measured": "tracked speed when 20 yards are covered", "physical": "speed at the middle of the run", "vs_official": "no official split isolates it",
                   "football": PHASES["middle"]["relevance"], "next": "film of intermediate movements", "cannot": "whether the speed is reached in pads, in a game"},
    "speed_30yd": {"measured": "tracked speed when 30 yards are covered", "physical": "speed late in the run", "vs_official": "close to the official forty for most athletes",
                   "football": PHASES["late"]["relevance"], "next": "film of long movements", "cannot": "top speed beyond the recording"},
    "fingerprint": {"measured": "distance between two athletes in the TCN-B representation of the forty sequence", "physical": "none: a mathematical pattern of the whole recording",
                    "vs_official": "almost unrelated to the official times, and highly repeatable between attempts", "football": "not interpretable physically",
                    "next": "a prompt to look at the two speed curves side by side", "cannot": "an athletic trait: no dimension has a physical meaning, and repeatable is not predictive"},
    "coverage": {"measured": "share of observed NFL episodes of a demand (same position) at or below the prospect's measured Combine value",
                 "physical": "where a maximal-effort Combine value sits inside the range of game demands", "vs_official": "uses tracked band values the official tests do not isolate",
                 "football": "the named movement demand, in physical terms", "next": "film of that demand; a targeted test of that band",
                 "cannot": "that the prospect will meet the demand in a game: descriptive similarity, not a validated ability"},
}


def assert_language(text: str) -> None:
    """Raise when a generated sentence uses a grade, ranking label or causal promise."""
    low = text.lower()
    bad = [p for p in FORBIDDEN_PHRASES if p in low]
    if bad:
        raise ValueError(f"unsupported wording {bad} in generated text: {text[:160]!r}")


def fmt(x: Any, unit: str = "", digits: int = 2, signed: bool = False) -> str:
    if x is None or (isinstance(x, float) and x != x):
        return "not recorded"
    s = f"{x:+.{digits}f}" if signed else f"{x:.{digits}f}"
    return f"{s} {unit}".strip()


def interpretation_registry(evidence: Iterable[dict[str, Any]], official: bool = True) -> list[dict[str, str]]:
    """One row per displayed metric with the seven answers. ``evidence``: rows of ``movement_matching.EVIDENCE``."""
    ev = {r["feature"]: r for r in evidence}
    rows = []
    for key, t in INTERPRETATIONS.items():
        e = ev.get(key)
        if e is not None:
            icc = "n/a" if e.get("icc_within_position") is None else f"{e['icc_within_position']:.2f}"
            r2 = "n/a" if e.get("r2_official_times") is None else f"{e['r2_official_times']:.2f}"
            rel = f"{e['evidence_status']} — repeat-attempt ICC {icc} ({e['n_repeat']} athletes with a repeat); R² on official times {r2} ({e['evidence_source']})"
        elif key in ("cmb_forty", "cmb_ten_yd_split"):
            rel = "single official value: measurement error not estimable from this data"
        elif key == "fingerprint":
            rel = "repeat-attempt retrieval is strong (Milestones 04B and 08B.4); a similarity, not a measurement"
        else:
            rel = "inherits the repeat-attempt error of the Combine value; shown with an interval or abstains"
        if key.startswith("cmb_") and not official:
            continue
        rows.append({"metric": key, "what_is_measured": t["measured"], "physical_meaning": t["physical"], "difference_from_official_testing": t["vs_official"],
                     "reliability": rel, "related_football_movements": t["football"] + " (physical relation only)", "what_to_look_at_next": t["next"],
                     "what_cannot_be_concluded": t["cannot"]})
    return rows


# ============================================================================== pair reading
def _usable(d: dict[str, Any]) -> bool:
    return d.get("evidence_status") not in ("unstable", "insufficient_support", None)


def _leaders(diffs: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    got = [d for d in diffs if d.get("verdict") == "distinguishable" and _usable(d)]
    return sorted(got, key=lambda d: -abs(d["z"]))


def pair_reading(twin: dict[str, Any], diffs: Sequence[dict[str, Any]]) -> dict[str, list[str]]:
    """Observation / Potential football relevance / Scouting follow-up / Limitations of one stopwatch-twin pair."""
    a, b, pos = twin["name_a"], twin["name_b"], twin["position_group"]
    lead = _leaders(diffs)
    obs = [f"{a} and {b} ({pos}) ran official forty times of {fmt(twin['forty_a'], 's')} and {fmt(twin['forty_b'], 's')} — {abs(twin['d_forty']):.2f} s apart."]
    if twin.get("d_ten") is not None:
        obs.append(f"Their official 10-yard splits are {fmt(twin['ten_a'], 's')} and {fmt(twin['ten_b'], 's')}.")
    for d in lead[:3]:
        who = b if d["quicker"] == "b" else a
        obs.append(f"{d['label']}: {fmt(d['value_a'], d['unit'])} versus {fmt(d['value_b'], d['unit'])} — a difference of {fmt(d['difference'], d['unit'], signed=True)} "
                   f"[{d['ci_lower']:+.2f}, {d['ci_upper']:+.2f}], larger than its measurement uncertainty. On this measure the observed value favours {who}.")
    soft = [d for d in diffs if d.get("verdict") == "not_distinguishable"]
    if not lead:
        obs.append("No tracked curve feature separates them beyond its measurement uncertainty: on the recorded evidence the two runs are not distinguishable.")
    elif soft:
        obs.append(f"{len(soft)} other feature(s) show a difference that is inside the measurement uncertainty and should not be read as a difference.")
    phases = list(dict.fromkeys(d["phase"] for d in lead if d["phase"] in PHASES)) or []
    rel = [f"The measured difference sits in {PHASES[p]['label']}. That phase may be physically related to {PHASES[p]['relevance']}." for p in phases]
    rel.append(NOT_ESTABLISHED if phases else "With no distinguishable difference there is no movement phase to relate to football movements.")
    film = POSITION_FILM.get(pos, {})
    follow = [f"Film — compare both prospects on {film.get(p, PHASES[p]['relevance'])}." for p in phases]
    follow += [f"Testing — {PHASES[p]['test']}." for p in phases]
    if not phases:
        follow.append("No tracking-based follow-up is indicated; evaluate the two prospects on film and official testing as usual.")
    one = [n for n, k in ((a, "tier_a"), (b, "tier_b")) if twin.get(k) in ("low", "medium")]
    lim = ["An observed Combine difference is not demonstrated to cause or predict a football outcome.",
           "One drill, at maximal effort, from rest, without pads: game movement depends on role, opportunity and context."]
    if one:
        lim.append(f"Recording evidence is below the high tier for {', '.join(one)}: read the comparison with that in mind.")
    if any(d.get("verdict") == "insufficient_evidence" for d in diffs):
        lim.append("Some features carry no error estimate for this pair and are shown without a verdict.")
    out = {"Observation": obs, "Potential football relevance": rel, "Scouting follow-up": follow, "Limitations": lim}
    for lines in out.values():
        for s in lines:
            assert_language(s)
    return out


def case_views(case: dict[str, Any], twin: dict[str, Any], diffs: Sequence[dict[str, Any]], demand_rows: Sequence[dict[str, Any]] = (),
               nfl_rows: Sequence[dict[str, Any]] = ()) -> dict[str, dict[str, list[str]]]:
    """Scout, position-coach and general-manager views of one case. ``demand_rows``: ``movement_demands.FREQUENCY`` rows of
    the position · ``nfl_rows``: optional observed NFL movement of the two athletes (shown, never used to select)."""
    read = pair_reading(twin, diffs)
    lead = _leaders(diffs)
    phases = list(dict.fromkeys(d["phase"] for d in lead if d["phase"] in PHASES))
    freq = {r["demand_id"]: r for r in demand_rows}
    tested = [d for d in diffs if d.get("z") is not None]
    uncertain = [d["label"] for d in diffs if d.get("verdict") in ("not_distinguishable", "insufficient_evidence")]
    scout = {"What is different": read["Observation"],
             "Is it measurable": [f"{len(lead)} of {len(tested)} tested curve features differ by more than the uncertainty of the difference "
                                  "(repeat-attempt variation of both athletes)." if tested else "No feature could be tested for this pair."],
             "Film to examine": [s for s in read["Scouting follow-up"] if s.startswith("Film")] or ["No specific film prompt from tracking."],
             "What remains uncertain": ([f"Not separable on this evidence: {', '.join(uncertain[:5])}."] if uncertain else []) + read["Limitations"][:1]}
    demand = []
    for p in phases:
        r = freq.get(PHASES[p]["demand"])
        if r and r.get("supported"):
            demand.append(f"In tracked NFL games, {twin['position_group']} athletes show {r['episodes_per_100_plays']:.0f} '{r['demand_id']}' episodes per 100 eligible "
                          f"player-plays ({r['n_episodes']:,} episodes, {r['n_athletes']} athletes): the physical demand closest to {PHASES[p]['label']}.")
        elif r:
            demand.append(f"The NFL demand closest to {PHASES[p]['label']} ('{r['demand_id']}') has too few tracked episodes for this position to be described.")
    coach = {"Observed movement phase": [f"The measured difference sits in {PHASES[p]['label']}." for p in phases] or ["No movement phase is separated beyond measurement uncertainty."],
             "Relevant physical demand": demand or ["No NFL demand profile is attached to this comparison."],
             "Targeted movement testing": [s for s in read["Scouting follow-up"] if s.startswith("Testing")] or ["None indicated by tracking."],
             "Measurement limitations": ["A suggested test is a way to measure the phase again — its usefulness for football evaluation has not been established, "
                                         "and no training conclusion follows from an observed difference."] + read["Limitations"][1:]}
    new = [d for d in lead if "mostly_stopwatch" not in str(d.get("evidence_status")) and "redundant" not in str(d.get("evidence_status"))]
    dup = [d["label"] for d in lead if d not in new]
    justify = ("Yes — as a reason for additional film review or a targeted test of the named phase. It is not an input for ranking the two prospects."
               if new else "No — tracking adds nothing here that justifies additional evaluation beyond official testing and film.")
    gm = {"New information beyond official tests": [f"{d['label']} ({d['evidence_status']})" for d in new] or ["None that is both distinguishable and not already carried by the stopwatch."],
          "Strength of evidence": [f"{len(lead)} distinguishable feature(s) out of {len(tested)} tested; recording evidence tiers: {twin.get('tier_a') or 'n/a'} and {twin.get('tier_b') or 'n/a'}."],
          "Redundancy": [f"Largely the same information as official testing: {', '.join(dup)}."] if dup else ["No distinguishable feature of this pair is flagged as stopwatch-redundant."],
          "Uncertainty": ["Differences are tested against repeat-attempt variation only. NFL predictive value of any tracked Combine feature: not established."],
          "Does this justify additional evaluation": [justify]}
    if nfl_rows:
        shown = "; ".join(f"{r['name']}: {fmt(r.get('value'), r.get('unit', ''))} over {r.get('n_windows', 0)} windows" for r in nfl_rows)
        note = [f"Observed NFL movement, shown after the selection and not used for it — {shown}.",
                "Two athletes cannot show a relationship; this is an illustration of what was observed, in whichever direction it points."]
        scout["Observed NFL movement (post-draft, illustration)"] = note
        gm["Observed NFL movement (post-draft, illustration)"] = note
    out = {"scout": scout, "coach": coach, "general_manager": gm}
    for view in out.values():
        for lines in view.values():
            for s in lines:
                assert_language(s)
    return out


def views_rows(case_id: str, views: dict[str, dict[str, list[str]]]) -> list[dict[str, Any]]:
    """Flat rows (case_id, view, section, section_order, line_order, text) of ``case_views``."""
    return [{"case_id": case_id, "view": v, "section": sec, "section_order": i, "line_order": j, "text": s}
            for v, secs in views.items() for i, (sec, lines) in enumerate(secs.items()) for j, s in enumerate(lines)]


# ============================================================================== prospect report
def prospect_report(profile: dict[str, Any], metrics: Sequence[dict[str, Any]], twins: Sequence[dict[str, Any]], similarity: Sequence[dict[str, Any]],
                    evidence: Sequence[dict[str, Any]]) -> dict[str, list[str]]:
    """Sections of a one-prospect report. ``metrics``: ``movement_matching.PROSPECT`` rows of the athlete · ``twins``: his
    ranked twin rows (already oriented so that the other athlete is ``other_name``) · ``similarity``: his MATRIX rows."""
    name, pos = profile.get("display_name") or str(profile.get("nfl_id")), profile.get("position_group")
    ev = {r["feature"]: r for r in evidence}
    out: dict[str, list[str]] = {}
    out["Prospect summary"] = [f"{name} — {pos}, draft class {profile.get('draft_year') or 'n/a'}. Recording evidence tier: {profile.get('evidence_tier') or 'n/a'}"
                               + (f" ({profile['evidence_reasons']})." if profile.get("evidence_reasons") else ".")]
    out["Official testing"] = [f"Forty-yard dash {fmt(profile.get('official_forty'), 's')}; 10-yard split {fmt(profile.get('official_ten'), 's')}; "
                               f"weight {fmt(profile.get('weight'), 'lb', 0)}; height {fmt(profile.get('height'), 'in', 1)}."]
    track = []
    for m in metrics:
        if m.get("value") is None:
            continue
        se = "" if m.get("standard_error") is None else f" ± {1.96 * m['standard_error']:.2f}"
        pct = "" if m.get("pct_position") is None else f"; raw position percentile {m['pct_position']:.0f} of {m['n_reference']}"
        track.append(f"{m['label']}: {fmt(m['value'], m['unit'])}{se} over {m['n_attempts']} attempt(s){pct} — {ev.get(m['feature'], {}).get('evidence_status', 'n/a')}.")
    out["Tracking insights"] = track or ["No tracked forty attempt is available for this prospect: official testing only."]
    out["Matched prospects (stopwatch twins)"] = [
        f"{t['other_name']}: forty {abs(t['d_forty']):.2f} s apart; {t['n_distinguishable']} distinguishable curve feature(s)"
        + (f", largest in {t['top_feature']}" if t.get("top_feature") and t["n_distinguishable"] else "") + "." for t in twins[:5]] or ["No stopwatch twin within the caliper."]
    sim = []
    for s in similarity:
        if s.get("state") == "supported":
            sim.append(f"{s['demand_id']}: measured Combine value at or above {s['mean_coverage']:.0f} % of observed NFL episodes of this position "
                       f"[{s['coverage_lo']:.0f}, {s['coverage_hi']:.0f}] over {s['n_dimensions']} shared dimension(s) — descriptive similarity only.")
        elif s.get("state") == "unsupported_by_forty":
            sim.append(f"{s['demand_id']}: not described by the forty-yard dash — no statement.")
    out["Movement-demand correspondence"] = sim or ["No demand correspondence could be computed."]
    weak = sorted({ev[m["feature"]]["label"] for m in metrics if m["feature"] in ev and ev[m["feature"]]["evidence_status"] in ("unstable", "insufficient_support")})
    out["Evidence confidence"] = ["Every tracked value is shown with its repeat-attempt uncertainty, or without a verdict when that cannot be estimated."] \
        + ([f"Not reliable enough to describe an athlete: {', '.join(weak)}."] if weak else [])
    film = POSITION_FILM.get(pos, {})
    out["Suggested film-review checklist"] = [f"{PHASES[p]['label'].capitalize()} — {film.get(p, PHASES[p]['relevance'])}." for p in ("early", "middle", "late")]
    out["Suggested follow-up testing"] = [f"{PHASES[p]['label'].capitalize()} — {PHASES[p]['test']}." for p in ("early", "middle", "late")] \
        + ["These are ways to measure a phase again; their value for football evaluation has not been established."]
    out["Scientific limitations"] = [DISCLAIMER, "Percentiles are relative to the tracked Combine athletes of the same position in this dataset.",
                                     "A missing measurement is never imputed and never read as a low value."]
    for lines in out.values():
        for s in lines:
            assert_language(s)
    return out


# ============================================================================== HTML
CSS = """
body{margin:0;background:#0b1626;color:#e8edf4;font:15px/1.5 system-ui,-apple-system,'Segoe UI',sans-serif}
.wrap{max-width:980px;margin:0 auto;padding:28px 28px 48px}
header{border-bottom:3px solid #c9a227;padding-bottom:14px;margin-bottom:22px}
.kicker{letter-spacing:.14em;text-transform:uppercase;font-size:12px;color:#c9a227}
h1{font-size:30px;margin:4px 0 2px}h2{font-size:17px;margin:26px 0 8px;color:#9ec5f4;border-left:4px solid #c9a227;padding-left:10px}
h3{font-size:14px;margin:14px 0 4px;color:#c3c2b7;text-transform:uppercase;letter-spacing:.06em}
ul{margin:4px 0 0 0;padding-left:20px}li{margin:3px 0}
.note{background:#12233b;border:1px solid #23395a;border-radius:8px;padding:12px 14px;color:#c3cede;font-size:13px}
.fig{background:#12233b;border:1px solid #23395a;border-radius:8px;padding:10px;margin:12px 0}.fig p{margin:6px 0 0;font-size:12px;color:#9fb0c7}
table{border-collapse:collapse;width:100%;font-size:13px}th,td{border-bottom:1px solid #23395a;padding:6px 8px;text-align:left}th{color:#9ec5f4}
footer{margin-top:30px;font-size:12px;color:#8fa1ba}
@media print{body{background:#fff;color:#111}.note,.fig{background:#f4f6f9;border-color:#ccd}h2{color:#0b1f3a}th{color:#0b1f3a}.kicker{color:#7a5f00}footer{color:#555}}
"""


def _table(rows: Sequence[dict[str, Any]], columns: Sequence[str]) -> str:
    cell = lambda v: "—" if v is None else (f"{v:.3f}" if isinstance(v, float) else html.escape(str(v)))
    head = "".join(f"<th>{html.escape(c)}</th>" for c in columns)
    body = "".join("<tr>" + "".join(f"<td>{cell(r.get(c))}</td>" for c in columns) + "</tr>" for r in rows)
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def render_html(title: str, subtitle: str, sections: dict[str, Any], figures: Sequence[tuple[str, str]] = (), tables: Sequence[tuple[str, Sequence[dict[str, Any]], Sequence[str]]] = (),
                footer: str = "") -> str:
    """Self-contained HTML report. ``sections``: heading → lines, or heading → {sub-heading → lines}. ``figures``:
    (caption, HTML or SVG fragment) — inserted as given, so the caller decides whether a script is embedded."""
    parts = [f"<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>"
             f"<title>{html.escape(title)}</title><style>{CSS}</style></head><body><div class='wrap'><header><div class='kicker'>Combine Intelligence Lab · "
             f"same stopwatch, different movement, better questions</div><h1>{html.escape(title)}</h1><div>{html.escape(subtitle)}</div></header>",
             f"<div class='note'>{html.escape(DISCLAIMER)}</div>"]
    ul = lambda lines: "<ul>" + "".join(f"<li>{html.escape(s)}</li>" for s in lines) + "</ul>"
    for head, body in sections.items():
        parts.append(f"<h2>{html.escape(head)}</h2>")
        if isinstance(body, dict):
            parts += [f"<h3>{html.escape(k)}</h3>{ul(v)}" for k, v in body.items()]
        else:
            parts.append(ul(body))
    for caption, frag in figures:
        parts.append(f"<div class='fig'>{frag}<p>{html.escape(caption)}</p></div>")
    for caption, rows, cols in tables:
        parts.append(f"<h2>{html.escape(caption)}</h2>{_table(rows, cols)}")
    parts.append(f"<footer>{html.escape(footer)}</footer></div></body></html>")
    return "".join(parts)


__all__ = ["DISCLAIMER", "FORBIDDEN_PHRASES", "INTERPRETATIONS", "NOT_ESTABLISHED", "PHASES", "POSITION_FILM", "VERSION", "assert_language", "case_views",
           "fmt", "interpretation_registry", "pair_reading", "prospect_report", "render_html", "views_rows"]
