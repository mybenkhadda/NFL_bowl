"""Milestone 10B — three signature football stories, selected by declared rules from the Milestone 09 application mart.

Standard library only (mart rows as dicts), shared by notebook 10B and the application's story gallery.

    STORY 1  WR — same speed, different development        ``select_wr_pair`` (criteria 1–6 below; the pair named in the
                                                            brief is used only if it passes every criterion)
    STORY 2  Same forty, different curve                   ``select_same_forty`` (the 09D case of that type, validated)
    STORY 3  The position-specific movement gap            ``movement_gap.gap_position`` (declared exposure rule)

    pair_story / gap_story   every deliverable of a story: hero caption, comparison card, ≤ 100-word explanation,
                             technical interpretation, scouting checklist, 30-second narration, provenance, fallback

No NFL outcome or NFL exposure value enters any selection (the inputs are pre-draft twin rows and differences). Stories
never say which athlete is the better NFL player: the wording guard of ``scouting_interpretation`` applies to every line.
"""

from __future__ import annotations

from typing import Any, Sequence

from . import movement_gap as mg
from . import product_kit as pk
from .scouting_interpretation import NOT_ESTABLISHED, PHASES, POSITION_FILM, assert_language, fmt

VERSION = "10.1"
WR_PREFERRED = ("Ladd McConkey", "Isaiah Bond")
SAME_FORTY_PREFERRED = ("Anthony Bradford", "Mason McCormick")
WR_CRITERIA = (
    ("valid_match", "stored stopwatch twins with the 10-yard split also inside the caliper"),
    ("feature_evidence", "≥ 1 distinguishable feature that is repeatable and not flagged stopwatch-redundant"),
    ("uncertainty_support", "≥ 2 features whose difference exceeds 1.96 standard errors of the difference, on repeatable features"),
    ("attempts", "≥ 2 tracked attempts for both athletes"),
    ("physical_interval", "a distinguishable difference in mid-sprint or late speed development (09B phase middle or late)"),
    ("contrast", "ranked by the largest |z| of a usable distinguishable difference (clear visual contrast)"),
)
MIN_ATTEMPTS = 2
NARRATION_WORDS = 80
SUMMARY_WORDS = 100


def _lead(diffs: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted((d for d in diffs if d.get("verdict") == "distinguishable" and pk.usable(d) and d.get("z") is not None), key=lambda d: -abs(float(d["z"])))


def _redundant(d: dict[str, Any]) -> bool:
    s = str(d.get("evidence_status"))
    return "mostly_stopwatch" in s or "redundant" in s


# ============================================================================== selection
def wr_candidates(twins: Sequence[dict[str, Any]], diffs_by_pair: dict[str, Sequence[dict[str, Any]]], profiles: dict[int, dict[str, Any]],
                  position: str = "WR") -> list[dict[str, Any]]:
    """Every stored twin pair of ``position`` with the six criteria evaluated (True / False) and its contrast."""
    out = []
    for t in twins:
        if t.get("position_group") != position:
            continue
        diffs = diffs_by_pair.get(t["pair_id"], [])
        lead = _lead(diffs)
        att = [int((profiles.get(t[k]) or {}).get("n_tracked_attempts") or 0) for k in ("nfl_id_a", "nfl_id_b")]
        c = {"valid_match": bool(t.get("ten_matched")), "feature_evidence": any(not _redundant(d) for d in lead), "uncertainty_support": len(lead) >= 2,
             "attempts": min(att) >= MIN_ATTEMPTS, "physical_interval": any(d.get("phase") in ("middle", "late") for d in lead)}
        out.append({"pair_id": t["pair_id"], "name_a": t["name_a"], "name_b": t["name_b"], "forty_a": t.get("forty_a"), "forty_b": t.get("forty_b"),
                    "attempts_a": att[0], "attempts_b": att[1], "n_distinguishable_usable": len(lead), **c, "eligible": all(c.values()),
                    "contrast": abs(float(lead[0]["z"])) if lead else 0.0, "top_feature": lead[0]["feature"] if lead else None,
                    "preferred": {t["name_a"], t["name_b"]} == set(WR_PREFERRED)})
    return sorted(out, key=lambda r: (-int(r["eligible"]), -r["contrast"], r["pair_id"]))


def select_wr_pair(candidates: Sequence[dict[str, Any]]) -> tuple[dict[str, Any] | None, str]:
    """(selected candidate, why). The preferred pair wins only if it passes every criterion; otherwise the eligible pair
    with the largest contrast. None when no pair is eligible (the story then shows its fallback)."""
    pref = next((c for c in candidates if c["preferred"]), None)
    if pref and pref["eligible"]:
        return pref, f"the pair named in the brief ({' and '.join(WR_PREFERRED)}) passes all six criteria"
    best = next((c for c in candidates if c["eligible"]), None)
    if pref and not pref["eligible"]:
        failed = [k for k, _ in WR_CRITERIA[:5] if not pref[k]]
        why = f"the pair named in the brief fails {', '.join(failed)}; "
    else:
        why = "the pair named in the brief is not a stored twin pair; " if not pref else ""
    if best is None:
        return None, why + "no WR twin pair passes all criteria"
    return best, why + f"selected the eligible WR pair with the largest contrast (|z| {best['contrast']:.2f})"


def select_same_forty(cases: Sequence[dict[str, Any]], twins: dict[str, dict[str, Any]]) -> tuple[dict[str, Any] | None, str]:
    """The 09D case of type ``same_forty_different_curve`` naming the preferred pair, validated as: identical official
    forty (to the hundredth) and ≥ 1 distinguishable feature. Otherwise the first validated case of that type."""
    pool = [c for c in cases if c.get("case_type") == "same_forty_different_curve" and not c.get("selected_using_nfl_outcome")]
    ok = lambda c: (t := twins.get(c["pair_id"])) is not None and t.get("forty_a") is not None and t.get("forty_b") is not None \
        and round(float(t["forty_a"]), 2) == round(float(t["forty_b"]), 2) and int(t.get("n_distinguishable") or 0) >= 1
    pref = next((c for c in pool if {c["name_a"], c["name_b"]} == set(SAME_FORTY_PREFERRED)), None)
    if pref and ok(pref):
        return pref, "the 09D case named in the brief: identical official forty and ≥ 1 distinguishable feature"
    alt = next((c for c in pool if ok(c)), None)
    if alt:
        return alt, "the named case did not validate; first 09D case of the type with an identical official forty"
    return None, "no 09D case of type same_forty_different_curve has an identical official forty"


# ============================================================================== story content
def _claims(claims: dict[str, dict[str, Any]], ids: Sequence[str]) -> list[str]:
    return [f"{k}: {c['claim']} — {c['value']}" for k in ids if (c := claims.get(k)) and c.get("executed")]


def _demand(pos: str, ph: str, demands: dict[str, dict[str, Any]]) -> str | None:
    r = demands.get(PHASES[ph]["demand"])
    if r and r.get("supported"):
        return (f"In tracked NFL games, {pos} athletes show {float(r['episodes_per_100_plays']):.1f} '{r.get('label', r['demand_id'])}' episodes per 100 eligible "
                f"player-plays ({int(r['n_episodes']):,} episodes, {int(r['n_athletes'])} athletes).")
    return None


def pair_story(kind: str, twin: dict[str, Any], diffs: Sequence[dict[str, Any]], profiles: dict[int, dict[str, Any]], demands: Sequence[dict[str, Any]],
               claims: dict[str, dict[str, Any]], provenance: dict[str, Any]) -> dict[str, Any]:
    """Every deliverable of a pair story. ``twin`` / ``diffs``: the stored (unoriented) 09C row and differences."""
    a, b, pos = twin["name_a"], twin["name_b"], twin["position_group"]
    pa, pb = profiles.get(twin["nfl_id_a"], {}), profiles.get(twin["nfl_id_b"], {})
    dem = {r["demand_id"]: r for r in demands if r.get("position_group") == pos}
    lead = _lead(diffs)
    tested = [d for d in diffs if d.get("z") is not None]
    phases = list(dict.fromkeys(d["phase"] for d in lead if d.get("phase") in PHASES and d["phase"] != "whole"))
    decomp = pk.phase_decomposition(diffs)
    same = round(float(twin["forty_a"]), 2) == round(float(twin["forty_b"]), 2)
    clock = f"identical official forties of {twin['forty_a']:.2f} s" if same else f"official forties of {twin['forty_a']:.2f} s and {twin['forty_b']:.2f} s"
    title = {"wr": f"Same speed, different development — {a} and {b} ({pos})", "same_forty": f"Same forty, different curve — {a} and {b} ({pos})"}.get(kind, f"{a} and {b}")
    card = {"position": pos, "names": [a, b], "official_forty": [twin.get("forty_a"), twin.get("forty_b")], "official_ten": [twin.get("ten_a"), twin.get("ten_b")],
            "weight": [pa.get("weight"), pb.get("weight")], "tracked_attempts": [pa.get("n_tracked_attempts"), pb.get("n_tracked_attempts")],
            "evidence_tier": [twin.get("tier_a"), twin.get("tier_b")], "distinguishable": f"{len(lead)} of {len(tested)} tested features",
            "phases": [PHASES[p]["label"] for p in phases],
            "largest": None if not lead else f"{lead[0]['label']}: {fmt(lead[0]['value_a'], lead[0]['unit'])} vs {fmt(lead[0]['value_b'], lead[0]['unit'])}"}
    s1 = f"{a} and {b} ({pos}) ran {clock}."
    if lead:
        d = lead[0]
        s2 = (f"Their 10 Hz tracking separates them on {len(lead)} of {len(tested)} measured features beyond measurement uncertainty; the largest is {d['label']}: "
              f"{fmt(d['value_a'], d['unit'])} versus {fmt(d['value_b'], d['unit'])}.")
    else:
        s2 = "Their 10 Hz tracking does not separate them beyond measurement uncertainty."
    s3 = f"The difference sits in {', '.join(PHASES[p]['label'] for p in phases[:2])}." if phases else ""
    s4 = (_demand(pos, phases[0], dem) or "") if phases else ""
    s5 = "That makes it a specific question for film and a targeted test — not a verdict on either player, because no tracked Combine feature has shown NFL predictive value here."
    summary = pk.fit_words([s for s in (s1, s2, s3, s5, s4) if s], SUMMARY_WORDS)
    tech = [f"Matching: stored 09C rule — same position, official forty within 0.03 s (|Δ| = {abs(float(twin['d_forty'])):.2f} s)"
            + ("" if twin.get("d_ten") is None else f", 10-yard split |Δ| = {abs(float(twin['d_ten'])):.2f} s") + "."]
    for d in lead:
        tech.append(f"{d['label']}: {fmt(d['value_a'], d['unit'])} vs {fmt(d['value_b'], d['unit'])}, difference {fmt(d['difference'], d['unit'], signed=True)} "
                    f"[{float(d['ci_lower']):+.2f}, {float(d['ci_upper']):+.2f}] (z = {float(d['z']):+.2f}); evidence status {d.get('evidence_status')}.")
    tech += [pk.phase_sentence(p, (a, b)) for p in decomp]
    red = [d["label"] for d in lead if _redundant(d)]
    if red:
        tech.append(f"Largely the same information as official testing: {', '.join(red)}.")
    tech.append("Uncertainty: each interval is ±1.96 standard errors of the difference built from both athletes' repeat-attempt variation; about 5 % of features "
                "would be called distinguishable by chance, and features of one run are correlated.")
    tech += _claims(claims, ("C02", "C04", "C08", "C14", "C17"))
    tech.append(NOT_ESTABLISHED)
    film = POSITION_FILM.get(pos, {})
    check = [f"Film — compare both on {film.get(p, PHASES[p]['relevance'])}." for p in phases] or ["Film — no tracking-based prompt; review as usual."]
    check += [f"Test — {PHASES[p]['test']}." for p in phases]
    check += ["Confirm both official times and the 10-yard split timing method.", "Check the recording tiers before quoting a tracked value."]
    if min(int(pa.get("n_tracked_attempts") or 0), int(pb.get("n_tracked_attempts") or 0)) < MIN_ATTEMPTS:
        check.append("Request a second tracked attempt: one athlete has a single recording.")
    nar = pk.fit_words([f"Two {pos} prospects. {clock.capitalize()}." if same else f"Two {pos} prospects, {clock}.",
                        "The stopwatch calls them the same." if same else "The stopwatch barely separates them.",
                        (f"Ten-hertz tracking does not: {lead[0]['label']}, {fmt(lead[0]['value_a'], lead[0]['unit'])} against {fmt(lead[0]['value_b'], lead[0]['unit'])}, "
                         "beyond measurement uncertainty.") if lead else "Tracking cannot separate them either.",
                        f"It sits in {PHASES[phases[0]]['label']}." if phases else "",
                        "The Lab turns that into a film question and a targeted test.", "It does not say who will play better."], NARRATION_WORDS)
    out = {"kind": kind, "title": title, "hero_caption": f"{a} vs {b}: official stopwatch, recorded 10 Hz speed curves, and each difference against its uncertainty.",
           "card": card, "summary_100": summary, "technical": tech, "checklist": check, "narration_30s": nar, "phases": decomp,
           "provenance": {**provenance, "pair_id": twin["pair_id"], "tables": ["stopwatch_twins", "twin_differences", "prospect_profiles", "nfl_movement_demands"],
                          "selected_using_nfl_outcome": False},
           "fallback": pk.NO_DATA["no_twin"]}
    for v in (out["title"], out["hero_caption"], out["summary_100"], out["narration_30s"], *out["technical"], *out["checklist"]):
        assert_language(v)
    return out


def gap_story(focus: dict[str, Any], matrix: Sequence[dict[str, Any]], claims: dict[str, dict[str, Any]], provenance: dict[str, Any]) -> dict[str, Any]:
    """Story 3 for the position chosen by ``movement_gap.gap_position``."""
    pos = focus["position_group"]
    rows = [r for r in matrix if r["position_group"] == pos]
    meas = [r for r in rows if r["coverage_class"] in (mg.DIRECT, mg.PARTIAL)]
    notm = sorted((r for r in rows if r["coverage_class"] == mg.NOT_MEASURED and r["supported"]), key=lambda r: -float(r["episodes_per_100_plays"] or 0))
    unav = [r for r in rows if r["coverage_class"] == mg.UNAVAILABLE]
    rate = lambda r: f"{float(r['episodes_per_100_plays'] or 0):.1f} per 100 eligible player-plays ({r['n_episodes']:,} episodes, {r['n_athletes']} athletes)"
    sections = {
        "What is measured at the Combine?": ["One straight-line sprint from rest at maximal effort, timed (forty, 10-yard split) and tracked at 10 Hz."]
        + [f"{r['label']} — {r['class_label'].lower()}: {r['reason']}." for r in meas],
        "What is observed in NFL games?": [f"{r['label']}: {rate(r)}." for r in sorted(rows, key=lambda r: -float(r['episodes_per_100_plays'] or 0)) if r["supported"]],
        "What is not measured by the forty?": [f"{r['label']}: {rate(r)} — {r['reason']}." for r in notm]
        + [f"{r['label']}: {r['reason']}." for r in unav],
        "Which follow-up measurement would fit better?": [f"{p['test']} — {p['protocol']} (addresses {p['demand']})." for p in mg.proposals(matrix, pos)] + [mg.LIMIT],
        "What can the scout conclude now?": [
            f"{100 * focus['share_not_measured']:.0f} % of the {pos} exposure index falls in demands the forty does not physically describe.",
            "The forty remains a valid description of straight-line acceleration from rest; it is silent on braking, redirection and re-acceleration.",
            "Those demands must be judged on film or measured by a targeted test until a validated measurement exists.", NOT_ESTABLISHED]}
    top = notm[0] if notm else None
    summary = pk.fit_words([
        f"For {pos}, {100 * focus['share_not_measured']:.0f} % of observed NFL movement exposure falls in demands the forty-yard dash cannot physically describe.",
        (f"The most frequent is {top['label']}, at {float(top['episodes_per_100_plays']):.1f} episodes per 100 eligible player-plays.") if top else "",
        "The forty describes straight-line acceleration from rest well; braking, redirection and re-acceleration are outside it.",
        "Scouts should judge those on film now, and a short tracked deceleration or re-acceleration test is the candidate measurement — its value is unproven."], SUMMARY_WORDS)
    nar = pk.fit_words([f"Now the position view: {pos}.", f"{100 * focus['share_not_measured']:.0f} percent of their observed game movement exposure is in demands the forty never asks for.",
                        f"{top['label'].capitalize()} alone: {float(top['episodes_per_100_plays']):.0f} episodes per hundred plays." if top else "",
                        "The forty is not wrong. It is incomplete.", "So the Lab names the gap and proposes a test, without claiming that test predicts anything."], NARRATION_WORDS)
    check = [f"Film — {pos} {r['label']} situations ({rate(r)})." for r in notm[:3]] + [f"Test proposal — {p['test']}." for p in mg.proposals(matrix, pos)[:3]]
    out = {"kind": "gap", "title": f"The movement gap — what the forty cannot see for {pos}",
           "hero_caption": f"{pos}: observed NFL movement demands (episodes per 100 eligible player-plays) coloured by what forty tracking can describe.",
           "card": {"position": pos, "share_not_measured": focus["share_not_measured"], "rate_total": focus["rate_total"], "n_eligible_plays": focus["n_eligible_plays"],
                    "top_unmeasured": None if not top else top["label"]},
           "summary_100": summary, "technical": [s for v in sections.values() for s in v] + _claims(claims, ("C09",)), "sections": sections,
           "checklist": check, "narration_30s": nar,
           "provenance": {**provenance, "position_group": pos, "tables": ["nfl_movement_demands", "nfl_demand_registry", "movement_matching_summary"],
                          "rule": "largest share of the summed exposure rate in demands without a forty analogue, among positions whose supported cells all have "
                                  f"≥ {mg.MIN_ATHLETES_COMPARE} athletes", "selected_using_nfl_outcome": False},
           "fallback": "No position has enough supported NFL demand cells for a gap story."}
    for v in (out["title"], out["hero_caption"], out["summary_100"], out["narration_30s"], *out["technical"], *out["checklist"]):
        assert_language(v)
    return out


def story_markdown(s: dict[str, Any], figure_rel: str | None = None) -> str:
    """One story as Markdown (used by notebook 10B and the submission builder)."""
    L = [f"## {s['title']}", ""]
    if figure_rel:
        L += [f"![{s['kind']}]({figure_rel})", "", f"*{s['hero_caption']}*", ""]
    L += ["**In 100 words.** " + s["summary_100"], "", "**30-second narration.** " + s["narration_30s"], "", "**Comparison card**", ""]
    L += [f"- {k.replace('_', ' ')}: {v if not isinstance(v, list) else ' · '.join('—' if x is None else str(x) for x in v)}" for k, v in s["card"].items()]
    L += ["", "**Technical interpretation**", ""] + [f"- {t}" for t in s["technical"]]
    L += ["", "**Scouting follow-up checklist**", ""] + [f"- [ ] {c}" for c in s["checklist"]]
    p = s["provenance"]
    L += ["", "**Provenance.** " + "; ".join(f"{k}: {v}" for k, v in p.items()), "", f"**If the data are absent.** {s['fallback']}", ""]
    return "\n".join(L)


__all__ = ["MIN_ATTEMPTS", "NARRATION_WORDS", "SAME_FORTY_PREFERRED", "SUMMARY_WORDS", "VERSION", "WR_CRITERIA", "WR_PREFERRED", "gap_story", "pair_story",
           "select_same_forty", "select_wr_pair", "story_markdown", "wr_candidates"]
