"""Milestone 10D — Scouting Decision Simulator: a simulator of the scouting REVIEW workflow (not of a draft or a game).

Standard library only (mart rows as dicts in, structured text out), shared by notebook 10D and the application.

    QUESTIONS        the five scouting questions the workflow starts from
    simulate         question + one or two prospects → structured action:
                     Observation · Confidence · Potential football relevance · Next film review · Additional testing · Cannot conclude
    audience_sections    the same action for a scout, a position coach or a general manager
    render_report        self-contained HTML (the 09D renderer)
    sections_pdf         a text PDF of the same sections (matplotlib imported lazily; None when unavailable)

Every line is assembled by a deterministic rule from stored values and labelled with one evidence category:

    measured_difference      a difference larger than its repeat-attempt uncertainty on a repeatable feature
    uncertain_difference     a difference inside its uncertainty, or on a feature not repeatable enough to describe an athlete
    missing_observation      a value that is not recorded (never imputed, never read as low)
    measurement_mismatch     a game demand the forty does not physically describe
    additional_data_needed   what would have to be measured to answer the question

No probability of success, no ranking, no draft label is produced; ``scouting_interpretation.assert_language`` guards every line.
"""

from __future__ import annotations

import html as _html
from typing import Any, Sequence

from . import movement_gap as mg
from . import product_kit as pk
from .scouting_interpretation import DISCLAIMER, PHASES, POSITION_FILM, assert_language, fmt, render_html

VERSION = "10.1"
SECTIONS = ("Observation", "Confidence", "Potential football relevance", "Next film review", "Additional testing", "Cannot conclude")
CATEGORIES = {"measured_difference": "Measured difference", "uncertain_difference": "Uncertain difference", "missing_observation": "Missing observation",
              "measurement_mismatch": "Measurement mismatch", "additional_data_needed": "Additional data needed", "context": "Context"}
QUESTIONS: dict[str, dict[str, Any]] = {
    "twins_differ": {"text": "Two prospects have similar forty times. What else differs?", "athletes": 2},
    "repeatable_advantage": {"text": "Is a tracked acceleration difference repeatable?", "athletes": 2},
    "demand_correspondence": {"text": "Which parts of a measured profile correspond to common NFL movement demands?", "athletes": 1},
    "not_measured": {"text": "Which relevant physical qualities does the forty not measure?", "athletes": 1},
    "follow_up_testing": {"text": "What follow-up testing would reduce uncertainty?", "athletes": 1},
}
ACCEL_FEATURES = ("acc_2_4", "acc_4_6", "acc_6_8", "time_4_8")
CANNOT = ("Which prospect will perform better in the NFL: no tracked Combine feature improved held-out prediction of NFL movement (Milestone 06) or "
          "receiver separation (Milestone 08B) beyond official testing.",
          "A draft order or a value judgement: the Lab describes measurements and does not rank prospects.",
          "That a physical similarity to an NFL demand means the demand will be met in pads, in traffic or late in a game.",
          "That a measured Combine difference causes a football outcome, or that training it would change one.")


class _Out:
    def __init__(self) -> None:
        self.sections: dict[str, list[dict[str, str]]] = {s: [] for s in SECTIONS}

    def add(self, section: str, category: str, text: str) -> None:
        assert_language(text)
        self.sections[section].append({"category": category, "text": text})


def _name(p: dict[str, Any] | None) -> str:
    return str((p or {}).get("display_name") or (p or {}).get("nfl_id") or "prospect")


def _official(out: _Out, p: dict[str, Any]) -> None:
    n = _name(p)
    out.add("Observation", "context", f"{n} ({p.get('position_group')}, class {p.get('draft_year') or 'n/a'}): official forty {fmt(p.get('official_forty'), 's')}, "
            f"10-yard split {fmt(p.get('official_ten'), 's')}, weight {fmt(p.get('weight'), 'lb', 0)}.")
    if p.get("official_ten") is None:
        out.add("Observation", "missing_observation", f"{n}: {pk.NO_DATA['missing_split']}")
    if int(p.get("n_tracked_attempts") or 0) == 0:
        out.add("Observation", "missing_observation", f"{n}: {pk.NO_DATA['no_tracked_attempt']}")


def _demand_line(pos: str, demand_id: str, demands: dict[str, dict[str, Any]], words: str) -> tuple[str, str]:
    r = demands.get(demand_id)
    if r and r.get("supported"):
        return "context", (f"{pos} athletes show {float(r['episodes_per_100_plays']):.1f} '{r.get('label', demand_id)}' episodes per 100 eligible player-plays "
                           f"({int(r['n_episodes']):,} episodes, {int(r['n_athletes'])} athletes) — the observed game demand physically closest to {words}.")
    return "missing_observation", f"The NFL demand closest to {words} ('{demand_id}'): {pk.NO_DATA['no_nfl_observation']}"


def _single(out: _Out, p: dict[str, Any], metrics: Sequence[dict[str, Any]], evidence: dict[str, dict[str, Any]], focus: Sequence[str] = ()) -> list[str]:
    """Observation + Confidence for one prospect; returns the 09 phases with a recorded value."""
    n = _name(p)
    rows = [m for m in metrics if pk._num(m.get("value")) is not None and (not focus or m["feature"] in focus)]
    for m in rows:
        se = pk._num(m.get("standard_error"))
        pct = pk._num(m.get("pct_position"))
        iv = "" if se is None else f" ± {1.96 * se:.2f}"
        where = "" if pct is None else f"; raw position percentile {pct:.0f} of {m.get('n_reference') or 'n/a'} (for times, lower = quicker)"
        status = (evidence.get(m["feature"]) or {}).get("evidence_status")
        cat = "uncertain_difference" if status in pk.UNUSABLE else "context"
        out.add("Observation", cat, f"{n} — {m['label']}: {fmt(m['value'], m['unit'])}{iv}{where}.")
    st = pk.attempt_stability(rows)
    multi = [s for s in st if s["n_attempts"] >= 2]
    if rows and not multi:
        out.add("Confidence", "additional_data_needed", f"{n}: {pk.NO_DATA['single_attempt']} A second tracked attempt would allow it.")
    for s in multi[:4]:
        out.add("Confidence", "context", f"{n} — {s['label']}: {s['reading']}.")
    weak = sorted({(evidence.get(m['feature']) or {}).get('label', m['feature']) for m in rows if (evidence.get(m['feature']) or {}).get('evidence_status') in pk.UNUSABLE})
    if weak:
        out.add("Confidence", "uncertain_difference", f"Not repeatable enough to describe an athlete: {', '.join(weak)}.")
    tier = p.get("evidence_tier")
    if tier in ("low", "medium"):
        out.add("Confidence", "uncertain_difference", f"{n}: recording evidence tier '{tier}'. {pk.NO_DATA['low_confidence']}")
    elif tier:
        out.add("Confidence", "context", f"{n}: recording evidence tier '{tier}'.")
    return sorted({(evidence.get(m["feature"]) or {}).get("phase") for m in rows} - {None, "whole"})


def _pair(out: _Out, twin: dict[str, Any], diffs: Sequence[dict[str, Any]], focus: Sequence[str] = ()) -> list[str]:
    """Observation + Confidence for a stopwatch-twin pair (stored 09C differences); returns the 09 phases that differ."""
    a, b = twin["name_a"], twin["name_b"]
    rows = [d for d in diffs if not focus or d["feature"] in focus]
    out.add("Observation", "context", f"Matched by the stored rule: same position, official forty {abs(float(twin['d_forty'])):.2f} s apart"
            + ("" if twin.get("d_ten") is None else f", 10-yard split {abs(float(twin['d_ten'])):.2f} s apart") + ".")
    for p in pk.phase_decomposition(rows):
        if not p["features"]:
            if not focus:
                out.add("Observation", "missing_observation", f"{p['phase']}: {pk.NO_DATA['phase_not_identified']}")
            continue
        cat = {"distinguishable": "measured_difference", "not_distinguishable": "uncertain_difference"}.get(p["status"], "missing_observation")
        line = pk.phase_sentence(p, (a, b))
        if p["status"] == "distinguishable":
            r = p["lead"]
            who = b if r["quicker"] == "b" else a if r["quicker"] == "a" else ""
            if who:
                line += f" On this measure the observed value favours {who}."
        out.add("Observation", cat, line)
    tested = [d for d in rows if d.get("z") is not None]
    lead = [d for d in tested if d.get("verdict") == "distinguishable" and pk.usable(d)]
    out.add("Confidence", "context", f"{len(lead)} of {len(tested)} tested feature(s) differ by more than the uncertainty of the difference (repeat-attempt "
            f"variation of both athletes; about 5 % of features would by chance alone).")
    shaky = [d["label"] for d in rows if d.get("verdict") == "distinguishable" and not pk.usable(d)]
    if shaky:
        out.add("Confidence", "uncertain_difference", f"Distinguishable but on a feature not repeatable enough to describe an athlete: {', '.join(shaky)}.")
    none = [d["label"] for d in rows if d.get("verdict") == "insufficient_evidence"]
    if none:
        out.add("Confidence", "additional_data_needed", f"No repeat-attempt error estimate for {', '.join(none)}: a second tracked attempt is needed for a verdict.")
    for n, k in ((a, "tier_a"), (b, "tier_b")):
        if twin.get(k) in ("low", "medium"):
            out.add("Confidence", "uncertain_difference", f"{n}: recording evidence tier '{twin[k]}'. {pk.NO_DATA['low_confidence']}")
    redundant = [d["label"] for d in lead if "mostly_stopwatch" in str(d.get("evidence_status")) or "redundant" in str(d.get("evidence_status"))]
    if redundant:
        out.add("Confidence", "context", f"Largely the same information as official testing: {', '.join(redundant)}.")
    return list(dict.fromkeys(d["phase"] for d in sorted(lead, key=lambda d: -abs(d["z"])) if d.get("phase") in PHASES and d["phase"] != "whole"))


def _relevance(out: _Out, pos: str, phases: Sequence[str], demands: dict[str, dict[str, Any]]) -> None:
    for ph in phases:
        cat, line = _demand_line(pos, PHASES[ph]["demand"], demands, PHASES[ph]["label"])
        out.add("Potential football relevance", cat, line)
    if phases:
        out.add("Potential football relevance", "context", "A physical correspondence only: the football advantage of the measured difference has not been established.")


def _similarity(out: _Out, name: str, matrix: Sequence[dict[str, Any]], labels: dict[str, str]) -> None:
    for r in sorted(matrix, key=lambda r: str(r.get("demand_id"))):
        lab = labels.get(r["demand_id"], r["demand_id"])
        if r.get("state") == "supported" and pk._num(r.get("mean_coverage")) is not None:
            out.add("Potential football relevance", "context", f"{name} — {lab}: Combine value at or above {r['mean_coverage']:.0f} % of observed NFL episodes of the "
                    f"position [{r['coverage_lo']:.0f}, {r['coverage_hi']:.0f}] over {r.get('n_dimensions')} shared dimension(s). Descriptive similarity, not a prediction.")
        elif r.get("state") == "unsupported_by_forty":
            out.add("Potential football relevance", "measurement_mismatch", f"{lab}: {pk.NO_DATA['unsupported_demand']}")


def _gap(out: _Out, pos: str, gap: Sequence[dict[str, Any]], section: str = "Potential football relevance", top: int = 5) -> None:
    rows = sorted((r for r in gap if r["position_group"] == pos and r["coverage_class"] in (mg.NOT_MEASURED, mg.PARTIAL) and r["supported"]),
                  key=lambda r: -float(r["episodes_per_100_plays"] or 0.0))[:top]
    for r in rows:
        out.add(section, "measurement_mismatch", f"{r['label']} ({r['class_label'].lower()}): {float(r['episodes_per_100_plays']):.1f} episodes per 100 eligible "
                f"{pos} player-plays ({r['n_athletes']} athletes) — {r['reason']}.")


def _film(out: _Out, pos: str, phases: Sequence[str], who: str) -> None:
    film = POSITION_FILM.get(pos, {})
    for ph in phases:
        out.add("Next film review", "additional_data_needed", f"Film — {who} on {film.get(ph, PHASES[ph]['relevance'])} ({PHASES[ph]['label']}).")
    if not phases:
        out.add("Next film review", "context", "No tracking-based film prompt: review the prospect(s) on film and official testing as usual.")


def _tests(out: _Out, phases: Sequence[str], pos: str, gap: Sequence[dict[str, Any]], with_gap: bool) -> None:
    for ph in phases:
        out.add("Additional testing", "additional_data_needed", f"{PHASES[ph]['label'].capitalize()} — {PHASES[ph]['test']}.")
    if with_gap:
        for pr in mg.proposals(gap, pos)[:3]:
            out.add("Additional testing", "additional_data_needed", f"{pr['test']} (for {pr['demand']}, {pr['rate']}): {pr['protocol']}.")
    out.add("Additional testing", "context", mg.LIMIT)


def simulate(question: str, a: dict[str, Any], metrics_a: Sequence[dict[str, Any]], *, b: dict[str, Any] | None = None,
             metrics_b: Sequence[dict[str, Any]] = (), twin: dict[str, Any] | None = None, diffs: Sequence[dict[str, Any]] = (),
             matrix_a: Sequence[dict[str, Any]] = (), demands: Sequence[dict[str, Any]] = (), registry: Sequence[dict[str, Any]] = (),
             evidence: Sequence[dict[str, Any]] = (), gap: Sequence[dict[str, Any]] = ()) -> dict[str, Any]:
    """The structured scouting action for ``question``.

    ``a`` / ``b``: ``prospect_profiles`` rows · ``metrics_*``: ``prospect_metrics`` rows · ``twin`` / ``diffs``: the stored
    stopwatch-twin row and its differences, oriented so that A is ``a`` (None when the two are not stored twins) ·
    ``matrix_a``: ``movement_similarity_matrix`` rows of ``a`` · ``demands``: ``nfl_movement_demands`` rows of the position ·
    ``registry``: ``nfl_demand_registry`` · ``evidence``: ``feature_evidence`` · ``gap``: ``movement_gap.gap_matrix`` rows."""
    if question not in QUESTIONS:
        raise KeyError(f"unknown question {question!r}; choose from {list(QUESTIONS)}")
    out = _Out()
    pos = str(a.get("position_group"))
    ev = {r["feature"]: r for r in evidence}
    dem = {r["demand_id"]: r for r in demands if r.get("position_group") == pos}
    labels = {r["demand_id"]: r.get("label", r["demand_id"]) for r in registry}
    pair = twin is not None and bool(diffs)
    _official(out, a)
    if b is not None:
        _official(out, b)
    focus = ACCEL_FEATURES if question == "repeatable_advantage" else ()
    if QUESTIONS[question]["athletes"] == 2 and b is None:
        out.add("Observation", "additional_data_needed", "Select a second prospect: this question compares two athletes.")
    if b is not None and not pair:
        out.add("Observation", "missing_observation", f"{_name(a)} and {_name(b)} are not a stored stopwatch-twin pair (same position, official forty within "
                "0.03 s): no difference with an uncertainty is available, so none is computed here.")
    if pair:
        phases = _pair(out, twin, diffs, focus)
        who = f"compare {twin['name_a']} and {twin['name_b']}"
    else:
        phases = _single(out, a, metrics_a, ev, focus) if int(a.get("n_tracked_attempts") or 0) > 0 else []
        if b is not None and int(b.get("n_tracked_attempts") or 0) > 0:
            _single(out, b, metrics_b, ev, focus)
        who = f"review {_name(a)}"
    if question == "repeatable_advantage":
        for f in ACCEL_FEATURES:
            e = ev.get(f)
            lab = pk.evidence_label(e)
            out.add("Confidence", "uncertain_difference" if (e or {}).get("evidence_status") in pk.UNUSABLE else "context",
                    f"{(e or {}).get('label', f)} as a measurement: {lab['repeatability']}; {lab['redundancy']}.")
    if question in ("twins_differ", "repeatable_advantage", "demand_correspondence", "follow_up_testing"):
        _relevance(out, pos, phases, dem)
    if question == "demand_correspondence":
        _similarity(out, _name(a), matrix_a, labels)
    if question in ("not_measured", "follow_up_testing", "demand_correspondence"):
        _gap(out, pos, gap)
    if question == "not_measured" and not gap:
        out.add("Potential football relevance", "missing_observation", "No movement-gap rows were passed for this position.")
    _film(out, pos, phases, who)
    _tests(out, phases, pos, gap, with_gap=question in ("not_measured", "follow_up_testing"))
    for c in CANNOT:
        out.add("Cannot conclude", "context", c)
    lead = [x for x in out.sections["Observation"] if x["category"] == "measured_difference"]
    mism = [x for s in out.sections.values() for x in s if x["category"] == "measurement_mismatch"]
    if lead:
        justify = "Additional evaluation is supported — as film review and a targeted test of the named phase(s), not as an input for ranking."
    elif mism and question in ("not_measured", "follow_up_testing"):
        justify = "The forty does not describe these demands: a targeted test is the way to observe them; the Combine data alone cannot."
    else:
        justify = "Tracking adds no measured difference here: evaluate on film and official testing as usual."
    assert_language(justify)
    counts = {k: sum(x["category"] == k for s in out.sections.values() for x in s) for k in CATEGORIES}
    return {"question_id": question, "question": QUESTIONS[question]["text"], "athletes": [_name(a)] + ([_name(b)] if b is not None else []),
            "position_group": pos, "pair_id": (twin or {}).get("pair_id"), "sections": out.sections, "decision": justify, "category_counts": counts,
            "engine_version": VERSION}


def orient_pair(twin: dict[str, Any], diffs: Sequence[dict[str, Any]], nfl_id: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """A stored twin row and its differences with ``nfl_id`` as athlete A (differences keep the convention B − A).
    Same rule as the application's ``data_access.orient_twin`` / ``differences``; nothing is recomputed."""
    if twin["nfl_id_a"] == nfl_id:
        return {**twin, "flip": False}, [dict(d) for d in diffs]
    neg = lambda x: None if x is None else -x
    t = {**twin, "flip": True}
    for k in ("nfl_id", "name", "draft_year", "forty", "ten", "tier"):
        t[f"{k}_a"], t[f"{k}_b"] = twin.get(f"{k}_b"), twin.get(f"{k}_a")
    for k in ("d_forty", "d_ten", "d_weight", "d_height"):
        t[k] = neg(twin.get(k))
    out = [{**d, "nfl_id_a": d.get("nfl_id_b"), "nfl_id_b": d.get("nfl_id_a"), "value_a": d.get("value_b"), "value_b": d.get("value_a"), "difference": neg(d.get("difference")),
            "ci_lower": neg(d.get("ci_upper")), "ci_upper": neg(d.get("ci_lower")), "z": neg(d.get("z")), "quicker": {"a": "b", "b": "a"}.get(d.get("quicker"), "")}
           for d in diffs]
    return t, out


# ============================================================================== audiences and rendering
def audience_sections(result: dict[str, Any], audience: str) -> dict[str, list[str]]:
    """``scout`` (everything) · ``coach`` (movement phases, film, testing) · ``gm`` (decision, evidence strength, limits)."""
    S = result["sections"]
    tag = lambda x: f"[{CATEGORIES[x['category']]}] {x['text']}" if x["category"] != "context" else x["text"]
    if audience == "scout":
        out = {s: [tag(x) for x in S[s]] for s in SECTIONS}
    elif audience == "coach":
        out = {"Observed movement phases": [tag(x) for x in S["Observation"] if x["category"] != "context"] or ["No phase separated beyond uncertainty."],
               "Physical demand in games": [tag(x) for x in S["Potential football relevance"]] or ["No demand profile attached."],
               "Film to cut up": [x["text"] for x in S["Next film review"]], "Targeted testing": [x["text"] for x in S["Additional testing"]],
               "Limits": list(CANNOT[2:])}
    elif audience == "gm":
        meas = [x["text"] for x in S["Observation"] if x["category"] == "measured_difference"][:3]
        out = {"Decision": [result["decision"]], "What was measured": meas or ["No difference larger than its measurement uncertainty."],
               "Strength of evidence": [x["text"] for x in S["Confidence"]][:4],
               "Evidence categories": [f"{CATEGORIES[k]}: {v}" for k, v in result["category_counts"].items() if v and k != "context"],
               "Cannot conclude": [x["text"] for x in S["Cannot conclude"]]}
    else:
        raise KeyError(f"unknown audience {audience!r}: scout · coach · gm")
    for lines in out.values():
        for s in lines:
            assert_language(s)
    return out


AUDIENCE_TITLE = {"scout": "Scout report", "coach": "Position-coach report", "gm": "General-manager summary"}


def render_report(result: dict[str, Any], audience: str, footer: str, figures: Sequence[tuple[str, str]] = ()) -> str:
    title = f"{AUDIENCE_TITLE[audience]} — {' vs '.join(result['athletes'])}"
    sub = f"{result['position_group']} · scouting question: {result['question']}"
    return render_html(title, sub, audience_sections(result, audience), figures, (), footer)


def sections_pdf(title: str, subtitle: str, sections: dict[str, list[str]], footer: str = "") -> bytes | None:
    """A paginated text PDF (US Letter) of report sections. None when matplotlib is not importable."""
    try:
        import io
        import textwrap

        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages
    except Exception:                                                     # noqa: BLE001 — the HTML report stays available
        return None
    lines: list[tuple[str, str]] = [("title", title), ("sub", subtitle), ("note", DISCLAIMER), ("gap", "")]
    for head, body in sections.items():
        lines.append(("head", head))
        for s in body:
            wrapped = textwrap.wrap(s, 105) or [""]
            lines += [("item", ("• " if k == 0 else "  ") + w) for k, w in enumerate(wrapped)]
        lines.append(("gap", ""))
    style = {"title": (15, "bold", "#0b1f3a"), "sub": (10, "normal", "#44546a"), "note": (8, "normal", "#44546a"), "head": (11, "bold", "#1f6f43"),
             "item": (9, "normal", "#111111"), "gap": (6, "normal", "#111111")}
    buf = io.BytesIO()
    with PdfPages(buf) as pdf:
        k = 0
        while k < len(lines):
            fig = plt.figure(figsize=(8.5, 11.0))
            y = 0.95
            while k < len(lines) and y > 0.06:
                kind, text = lines[k]
                size, weight, color = style[kind]
                if kind == "note":
                    for w in textwrap.wrap(text, 120):
                        fig.text(0.07, y, w, fontsize=size, color=color)
                        y -= 0.016
                else:
                    fig.text(0.07, y, text, fontsize=size, fontweight=weight, color=color)
                    y -= {"title": 0.03, "head": 0.026, "gap": 0.01}.get(kind, 0.018)
                k += 1
            fig.text(0.07, 0.03, footer, fontsize=7, color="#6b7a90")
            pdf.savefig(fig)
            plt.close(fig)
    return buf.getvalue()


def result_html_fragment(result: dict[str, Any]) -> str:
    """Compact HTML of the six sections (used inside the application cards)."""
    parts = []
    for s in SECTIONS:
        items = "".join(f"<li><span class='cat cat-{x['category']}'>{_html.escape(CATEGORIES[x['category']])}</span> {_html.escape(x['text'])}</li>"
                        if x["category"] != "context" else f"<li>{_html.escape(x['text'])}</li>" for x in result["sections"][s])
        parts.append(f"<h4>{_html.escape(s)}</h4><ul>{items or '<li>—</li>'}</ul>")
    return "".join(parts)


__all__ = ["ACCEL_FEATURES", "AUDIENCE_TITLE", "CANNOT", "CATEGORIES", "QUESTIONS", "SECTIONS", "VERSION", "audience_sections", "orient_pair", "render_report",
           "result_html_fragment", "sections_pdf", "simulate"]
