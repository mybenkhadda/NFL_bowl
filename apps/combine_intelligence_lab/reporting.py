"""Scouting reports of the application: self-contained HTML (prospect or pair) and a one-page PDF intelligence card.

The text comes from the shared interpretation engine (``scouting_interpretation``, the same code the notebooks use), so
the wording guard applies here too. Nothing is computed that the mart does not already hold.
"""

from __future__ import annotations

import io
from typing import Any

import numpy as np
import plotly.graph_objects as go

import data_access as da
from nflbdb.research.advanced_movement import scouting_interpretation as sin

COLS = ["label", "unit", "value_a", "value_b", "difference", "ci_lower", "ci_upper", "verdict", "evidence_status"]


def _fig_html(figs: list[tuple[str, go.Figure]]) -> list[tuple[str, str]]:
    """Plotly fragments; the library is embedded once (the report works offline)."""
    out = []
    for k, (caption, fig) in enumerate(figs):
        out.append((caption, fig.to_html(full_html=False, include_plotlyjs="inline" if k == 0 else False, config={"displaylogo": False})))
    return out


def prospect_sections(nfl_id: int) -> dict[str, list[str]]:
    prof = da.profile(nfl_id)
    twins = da.twins_of(nfl_id)
    matrix, _ = da.similarity(nfl_id)
    return sin.prospect_report(prof, da.metrics(nfl_id), twins, matrix, list(da.evidence_rows().values()))


def prospect_html(nfl_id: int, figs: list[tuple[str, go.Figure]]) -> str:
    prof = da.profile(nfl_id)
    title = f"{prof.get('display_name') or nfl_id}"
    sub = f"{prof.get('position_group')} · draft class {prof.get('draft_year') or 'n/a'} · prospect intelligence report"
    m = da.manifest()
    return sin.render_html(title, sub, prospect_sections(nfl_id), _fig_html(figs),
                           [("Tracked measures", da.metrics(nfl_id), ["label", "unit", "value", "standard_error", "n_attempts", "pct_position", "state"])],
                           f"Combine Intelligence Lab · mart {m.get('mart_version')} built {m.get('built_at')} · run mode {m.get('run_mode')}")


def pair_html(twin: dict[str, Any], diffs: list[dict[str, Any]], figs: list[tuple[str, go.Figure]], views: dict[str, dict[str, list[str]]] | None = None) -> str:
    sections: dict[str, Any] = {"Comparison": sin.pair_reading(twin, diffs)}
    if views:
        titles = {"scout": "Scout view", "coach": "Position-coach view", "general_manager": "General-manager view"}
        sections |= {titles.get(k, k): v for k, v in views.items()}
    m = da.manifest()
    return sin.render_html(f"{twin['name_a']} and {twin['name_b']}", f"Stopwatch twins · {twin['position_group']} · official forty {abs(twin['d_forty']):.2f} s apart",
                           sections, _fig_html(figs), [("Measured differences (B − A)", diffs, COLS)],
                           f"Combine Intelligence Lab · mart {m.get('mart_version')} built {m.get('built_at')} · run mode {m.get('run_mode')}")


def _footer() -> str:
    m = da.manifest()
    return f"Combine Intelligence Lab · mart {m.get('mart_version')} built {m.get('built_at')} · run mode {m.get('run_mode')}"


def decision_html(result: dict[str, Any], audience: str) -> str:
    """Scout report · position-coach report · general-manager summary of one simulator result (self-contained HTML)."""
    from nflbdb.research.advanced_movement import decision_simulator as ds

    return ds.render_report(result, audience, _footer())


def decision_pdf(result: dict[str, Any], audience: str) -> bytes | None:
    from nflbdb.research.advanced_movement import decision_simulator as ds

    title = f"{ds.AUDIENCE_TITLE[audience]} — {' vs '.join(result['athletes'])}"
    return ds.sections_pdf(title, f"{result['position_group']} · {result['question']}", ds.audience_sections(result, audience), _footer())


def card_pdf(nfl_id: int) -> bytes | None:
    """One-page PDF intelligence card (matplotlib, the notebook figure function). None when the project figure stack is
    not importable in this environment."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        from nflbdb.research.advanced_movement import intelligence_figures as ifg
    except Exception:                                    # noqa: BLE001 — the HTML report stays available
        return None
    prof = da.profile(nfl_id)
    att = da.curves(nfl_id)
    curves = [(np.asarray(a["dist"], float), np.asarray(a["speed"], float)) for a in att]
    b = da.band(prof.get("position_group") or "")
    band = {"grid": np.asarray(b["dist_yd"]), "q10": np.asarray(b["q10"]), "q50": np.asarray(b["q50"]), "q90": np.asarray(b["q90"]), "n": b["n"]} if b else None
    matrix, _ = da.similarity(nfl_id)
    fig = ifg.fig_intelligence_card(prof, da.metrics(nfl_id), da.twins_of(nfl_id), matrix, curves, band, prospect_sections(nfl_id))
    buf = io.BytesIO()
    fig.savefig(buf, format="pdf")
    import matplotlib.pyplot as plt

    plt.close(fig)
    return buf.getvalue()
