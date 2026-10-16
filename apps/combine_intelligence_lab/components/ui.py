"""Theme and reusable interface pieces: page chrome, stat tiles, evidence badges, the athlete picker, the disclaimer.

Design language: dark navy surface, gold yard-line accents, system sans, large readable figures. A status is always an
icon AND a label — never a colour alone. Text wears text colours; a coloured mark beside it carries identity.
"""

from __future__ import annotations

import html
from typing import Any, Sequence

import streamlit as st

import data_access as da

C = {"bg": "#0a1422", "panel": "#111d2e", "panel2": "#16263b", "border": "#22344d", "text": "#e8edf4", "muted": "#9fb0c7", "gold": "#c9a227",
     "field": "#2f9e5b", "field_dim": "rgba(47,158,91,.14)", "charcoal": "#14181f",
     "a": "#3987e5", "b": "#e0782f", "c": "#199e70", "good": "#2f9e5b", "warning": "#fab219", "critical": "#d03b3b", "neutral": "#8fa1ba", "grid": "#1c2c42"}
POSITION_COLOR = {"WR": "#3987e5", "DB": "#d95926", "TE": "#199e70", "DL": "#c98500", "OL": "#d55181", "ALL": "#8fa1ba"}
TAGLINE = "Same stopwatch. Different movement. Better questions."
DISCLAIMER = ("Descriptive scouting information. In this project, tracked Combine features did not improve held-out prediction of NFL movement or "
              "receiver separation beyond official testing. Similarity is not prediction; repeatability is not football value.")


def _css(present: bool) -> str:
    base = 18 if present else 15
    return f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [data-testid="stAppViewContainer"] {{ background: linear-gradient(180deg, {C['bg']} 0%, {C['charcoal']} 100%); color: {C['text']}; font-size: {base}px;
    font-family: 'Inter', system-ui, -apple-system, 'Segoe UI', sans-serif; }}
[data-testid="stHeader"] {{ background: transparent; }}
[data-testid="stSidebar"] {{ background: #070f1a; border-right: 1px solid {C['border']}; }}
.block-container {{ padding-top: 1.4rem; max-width: 1480px; }}
h1, h2, h3 {{ letter-spacing: -0.01em; font-weight: 700; }}
h3 {{ margin-top: 1.4rem; }}
.stButton > button[kind="primary"], .stDownloadButton > button[kind="primary"] {{ background: {C['field']}; border-color: {C['field']}; color: #fff; }}
[data-testid="stExpander"] {{ border: 1px solid {C['border']}; border-radius: 10px; background: {C['panel']}; }}
.cil-hero {{ border-bottom: 2px solid {C['field']}; padding: 0 0 14px; margin-bottom: 18px;
            background-image: repeating-linear-gradient(90deg, transparent 0 119px, rgba(47,158,91,.10) 119px 120px); }}
.cil-kicker {{ color: {C['field']}; letter-spacing: .16em; text-transform: uppercase; font-size: .72rem; font-weight: 700; }}
.cil-info {{ display: inline-block; width: 16px; height: 16px; line-height: 16px; text-align: center; border-radius: 50%; border: 1px solid {C['border']};
             color: {C['muted']}; font-size: .68rem; cursor: help; margin-left: 6px; }}
.cil-empty {{ border: 1px dashed {C['border']}; border-radius: 10px; padding: 18px; color: {C['muted']}; background: rgba(17,29,46,.6); margin: 8px 0 14px; }}
.cil-action h4 {{ margin: 12px 0 4px; font-size: .78rem; color: {C['field']}; text-transform: uppercase; letter-spacing: .1em; }}
.cil-action ul {{ margin: 0 0 6px; padding-left: 18px; }} .cil-action li {{ margin: 4px 0; }}
.cat {{ display: inline-block; font-size: .68rem; text-transform: uppercase; letter-spacing: .06em; padding: 1px 7px; border-radius: 4px; margin-right: 6px;
        border: 1px solid {C['border']}; color: {C['muted']}; }}
.cat-measured_difference {{ color: {C['field']}; border-color: {C['field']}; }}
.cat-uncertain_difference {{ color: {C['warning']}; border-color: {C['warning']}; }}
.cat-measurement_mismatch {{ color: {C['b']}; border-color: {C['b']}; }}
.cat-additional_data_needed {{ color: {C['a']}; border-color: {C['a']}; }}
.cil-story {{ background: {C['panel']}; border: 1px solid {C['border']}; border-top: 3px solid {C['field']}; border-radius: 10px; padding: 16px 18px; height: 100%; }}
.cil-story .k {{ color: {C['field']}; font-size: .72rem; letter-spacing: .14em; text-transform: uppercase; font-weight: 700; }}
.cil-story .t {{ font-size: 1.12rem; font-weight: 700; margin: 4px 0 8px; }}
.cil-story p {{ color: {C['muted']}; font-size: .9rem; }}
.cil-version {{ font-size: .72rem; color: {C['muted']}; border: 1px solid {C['border']}; border-radius: 6px; padding: 6px 8px; margin-top: 8px; }}
.cil-title {{ font-size: {2.5 if present else 2.0}rem; font-weight: 700; line-height: 1.15; margin: 2px 0 4px; }}
.cil-sub {{ color: {C['muted']}; font-size: 1rem; max-width: 70ch; }}
.cil-tiles {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 12px; margin: 6px 0 18px; }}
.cil-tile {{ background: {C['panel']}; border: 1px solid {C['border']}; border-radius: 10px; padding: 14px 16px; }}
.cil-tile .v {{ font-size: {2.3 if present else 1.9}rem; font-weight: 700; line-height: 1.1; }}
.cil-tile .l {{ color: {C['muted']}; font-size: .78rem; text-transform: uppercase; letter-spacing: .08em; margin-bottom: 4px; }}
.cil-tile .n {{ color: {C['muted']}; font-size: .8rem; margin-top: 4px; }}
.cil-card {{ background: {C['panel']}; border: 1px solid {C['border']}; border-radius: 10px; padding: 16px 18px; margin-bottom: 14px; }}
.cil-card h4 {{ margin: 0 0 8px; font-size: .8rem; color: {C['field']}; text-transform: uppercase; letter-spacing: .1em; }}
.cil-card ul {{ margin: 0; padding-left: 18px; }} .cil-card li {{ margin: 4px 0; }}
.cil-badge {{ display: inline-block; border: 1px solid {C['border']}; border-radius: 999px; padding: 2px 10px; font-size: .78rem; background: {C['panel2']};
              color: {C['text']}; white-space: nowrap; }}
.cil-dot {{ display: inline-block; width: 9px; height: 9px; border-radius: 50%; margin-right: 6px; }}
.cil-note {{ border-left: 3px solid {C['field']}; background: {C['panel']}; padding: 10px 14px; color: {C['muted']}; font-size: .86rem; border-radius: 0 8px 8px 0;
             margin: 10px 0 16px; }}
.cil-vs {{ display: grid; grid-template-columns: 1fr auto 1fr; gap: 14px; align-items: center; }}
.cil-vs .who {{ font-size: 1.25rem; font-weight: 700; }} .cil-vs .mid {{ color: {C['field']}; font-weight: 700; letter-spacing: .1em; }}
.cil-vs .t {{ font-size: 2rem; font-weight: 700; }}
table.cil {{ width: 100%; border-collapse: collapse; font-size: .88rem; }}
table.cil th {{ text-align: left; color: {C['muted']}; font-weight: 600; border-bottom: 1px solid {C['border']}; padding: 6px 8px; }}
table.cil td {{ border-bottom: 1px solid {C['grid']}; padding: 6px 8px; vertical-align: top; }}
</style>"""


def chrome(title: str, subtitle: str, kicker: str = "Combine Intelligence Lab") -> None:
    """Theme + page header. Call first on every page."""
    present = bool(st.session_state.get("present", False))
    st.markdown(_css(present), unsafe_allow_html=True)
    st.markdown(f"<div class='cil-hero'><div class='cil-kicker'>{html.escape(kicker)}</div><div class='cil-title'>{html.escape(title)}</div>"
                f"<div class='cil-sub'>{html.escape(subtitle)}</div></div>", unsafe_allow_html=True)
    m = da.manifest()
    if m.get("run_mode") == "smoke":
        st.warning("This mart was built in SMOKE mode: a wiring check on a subset of the data. The numbers are not results.", icon="⚠️")


def tiles(items: Sequence[tuple[str, Any, str]]) -> None:
    """Stat tiles: (label, value, note)."""
    body = "".join(f"<div class='cil-tile'><div class='l'>{html.escape(str(l))}</div><div class='v'>{html.escape(_num(v))}</div>"
                   f"<div class='n'>{html.escape(str(n))}</div></div>" for l, v, n in items)
    st.markdown(f"<div class='cil-tiles'>{body}</div>", unsafe_allow_html=True)


def _num(v: Any) -> str:
    if v is None:
        return "—"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, int):
        return f"{v:,}"
    if isinstance(v, float):
        return f"{v:.2f}"
    return str(v)


def fmt(v: Any, unit: str = "", d: int = 2) -> str:
    return "not recorded" if v is None else f"{v:.{d}f} {unit}".strip()


STATUS = (("repeatable_beyond_official", "good", "✔", "Repeatable · adds to official tests"),
          ("partly_beyond_stopwatch", "good", "✔", "Repeatable · partly beyond the stopwatch"),
          ("mostly_stopwatch", "neutral", "≈", "Repeatable · mostly the stopwatch"),
          ("redundant", "neutral", "≈", "Stopwatch-redundant"),
          ("weak_residual", "warning", "◐", "Weak additional information"),
          ("moderately", "warning", "◐", "Moderately repeatable"),
          ("unstable", "critical", "✖", "Not repeatable"),
          ("insufficient", "critical", "✖", "Insufficient evidence"))
VERDICT = {"distinguishable": ("good", "✔", "Distinguishable"), "not_distinguishable": ("neutral", "≈", "Not distinguishable"),
           "insufficient_evidence": ("warning", "◐", "No error estimate"), "not_observed": ("neutral", "–", "Not recorded")}


def badge(status: str | None) -> str:
    """Evidence badge (HTML): icon + label; the dot colour repeats the icon, it never stands alone."""
    for key, tone, icon, label in STATUS:
        if status and key in status:
            return f"<span class='cil-badge' title='{html.escape(status)}'><span class='cil-dot' style='background:{C[tone]}'></span>{icon} {label}</span>"
    return f"<span class='cil-badge'>{html.escape(status or 'n/a')}</span>"


def verdict_badge(v: str | None) -> str:
    tone, icon, label = VERDICT.get(v or "", ("neutral", "–", v or "n/a"))
    return f"<span class='cil-badge'><span class='cil-dot' style='background:{C[tone]}'></span>{icon} {label}</span>"


def card(title: str, lines: Sequence[str]) -> None:
    items = "".join(f"<li>{html.escape(s)}</li>" for s in lines) or "<li>—</li>"
    st.markdown(f"<div class='cil-card'><h4>{html.escape(title)}</h4><ul>{items}</ul></div>", unsafe_allow_html=True)


def note(text: str) -> None:
    st.markdown(f"<div class='cil-note'>{html.escape(text)}</div>", unsafe_allow_html=True)


def disclaimer() -> None:
    note(DISCLAIMER)


def html_table(rows: Sequence[dict[str, Any]], columns: Sequence[tuple[str, str]], raw: Sequence[str] = ()) -> None:
    """A styled table. ``columns``: (key, header). Keys in ``raw`` hold trusted HTML (badges); everything else is escaped."""
    cell = lambda r, k: (r.get(k) or "") if k in raw else html.escape("—" if r.get(k) is None else (f"{r[k]:.2f}" if isinstance(r[k], float) else str(r[k])))
    head = "".join(f"<th>{html.escape(h)}</th>" for _, h in columns)
    body = "".join("<tr>" + "".join(f"<td>{cell(r, k)}</td>" for k, _ in columns) + "</tr>" for r in rows)
    st.markdown(f"<table class='cil'><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>", unsafe_allow_html=True)


def versus(a: str, b: str, ta: float | None, tb: float | None, label: str = "official forty") -> None:
    f = lambda x: "—" if x is None else f"{x:.2f} s"
    st.markdown(f"<div class='cil-card'><h4>{html.escape(label)} — what the stopwatch sees</h4><div class='cil-vs'><div><div class='who'>{html.escape(a)}</div>"
                f"<div class='t'>{f(ta)}</div></div><div class='mid'>VS</div><div style='text-align:right'><div class='who'>{html.escape(b)}</div>"
                f"<div class='t'>{f(tb)}</div></div></div></div>", unsafe_allow_html=True)


# ============================================================================== Milestone 10 helpers
def info(tooltip: str) -> str:
    """An understated ⓘ with a hover tooltip (HTML) — used instead of a coloured badge where a badge would be noise."""
    return f"<span class='cil-info' title='{html.escape(tooltip, quote=True)}'>i</span>"


def evidence_info(feature: str) -> str:
    from nflbdb.research.advanced_movement import product_kit as pk

    return info(pk.evidence_label(da.evidence_rows().get(feature))["tooltip"])


def empty_state(kind: str, extra: str = "") -> None:
    """A declared no-data state (``product_kit.NO_DATA``). Never a placeholder value."""
    from nflbdb.research.advanced_movement import product_kit as pk

    st.markdown(f"<div class='cil-empty'>{html.escape(pk.NO_DATA.get(kind, kind))}{(' ' + html.escape(extra)) if extra else ''}</div>", unsafe_allow_html=True)


def action(result: dict[str, Any]) -> None:
    """The six-section structured scouting action of the decision simulator."""
    from nflbdb.research.advanced_movement import decision_simulator as ds

    st.markdown(f"<div class='cil-card cil-action'>{ds.result_html_fragment(result)}</div>", unsafe_allow_html=True)


def story_card(kicker: str, title: str, body: str) -> None:
    st.markdown(f"<div class='cil-story'><div class='k'>{html.escape(kicker)}</div><div class='t'>{html.escape(title)}</div><p>{html.escape(body)}</p></div>",
                unsafe_allow_html=True)


def version_badge() -> None:
    m = da.manifest()
    if not m:
        return
    from nflbdb.research.advanced_movement import product_kit as pk

    st.markdown(f"<div class='cil-version'>Data: mart <b>{html.escape(str(m.get('mart_version')))}</b> · {html.escape(str(m.get('run_mode')))} mode<br>"
                f"built {html.escape(str(m.get('built_at', ''))[:16])}<br>Lab engine {pk.VERSION}</div>", unsafe_allow_html=True)


def athlete_picker(label: str = "Prospect", tracked_only: bool = True, key: str = "athlete_id") -> int | None:
    """The shared athlete selection (one value for every page). Returns the nfl_id or None."""
    opts = da.athlete_labels(tracked_only=tracked_only, position=st.session_state.get("position_filter"))
    if not opts:
        st.info("No prospect matches the current position filter.")
        return None
    ids = list(opts)
    current = st.session_state.get("athlete_id")
    idx = ids.index(current) if current in ids else 0
    chosen = st.selectbox(label, ids, index=idx, format_func=lambda i: opts[i], key=f"{key}_widget", help="Type to search by name. The selection follows you across pages.")
    st.session_state["athlete_id"] = chosen
    return chosen
