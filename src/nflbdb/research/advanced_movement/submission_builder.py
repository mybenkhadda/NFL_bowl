"""Milestone 10E — submission builder: fills the authored submission documents from executed artifacts, and checks them.

The four documents of ``reports/milestone10/`` are authored once with two kinds of numbers:

* Milestone 09 numbers typed from ``reports/milestone09/EXECUTED_NUMBERS.md`` (executed 2026-10-14). ``check_static_numbers``
  re-reads each of them from the mart and reports any mismatch — the builder never edits them silently.
* Milestone 10 results as tokens ``‹M10:key›`` (story selections, movement-gap shares). ``render_documents`` keeps the authored
  file as a template under ``templates/`` on its first run and renders the filled document from it, so a rerun refreshes
  every value. An unfilled token stays visible and is listed in the audit.

    token_values          key → text, from the mart, the 10B stories and the 10C gap tables
    render_documents      template → filled document, with an audit table (tokens, filled, missing)
    check_static_numbers  the typed Milestone 09 numbers against the mart
    figure_index          the final figure set with its files on disk
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import Any, Sequence

import polars as pl

from ...data.utils import utc_now

VERSION = "10.1"
TOKEN = re.compile(r"‹M10:([a-z0-9_]+)›")
DOCUMENTS = ("NFL_BDB_2027_FINAL_SUBMISSION.md", "EXECUTIVE_SCOUTING_BRIEF.md", "TECHNICAL_APPENDIX.md", "SUBMISSION_WALKTHROUGH.md", "DEMO_VIDEO_STORYBOARD.md")
FINAL_FIGURES = (
    ("01", "Hero stopwatch-twin sprint comparison", "AM10E_01_hero_stopwatch_twins"),
    ("02", "Speed-development decomposition", "AM10E_02_phase_decomposition"),
    ("03", "Matched-pair distinguishability", "AM10E_03_twin_distinguishability"),
    ("04", "Reliability versus redundancy", "AM10E_04_reliability_vs_redundancy"),
    ("05", "NFL movement demands by position", "AM10E_05_nfl_demands_by_position"),
    ("06", "NFL Movement Gap matrix", "AM10E_06_movement_gap_matrix"),
    ("07", "Combine-to-NFL physical correspondence", "AM10E_07_physical_correspondence"),
    ("08", "One-player scouting profile", "AM10E_08_scouting_profile"),
    ("09", "Scout decision workflow", "AM10E_09_decision_workflow"),
    ("10", "Final contribution diagram", "AM10E_10_contribution"),
)


def _md_table(header: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    return "\n".join(["| " + " | ".join(header) + " |", "|" + "---|" * len(header)] + ["| " + " | ".join(str(x) for x in r) + " |" for r in rows])


def token_values(T: dict[str, pl.DataFrame], manifest: dict[str, Any], stories: dict[str, dict[str, Any]], gap_summary: Sequence[dict[str, Any]],
                 gap_focus: dict[str, Any] | None, gap_matrix: Sequence[dict[str, Any]]) -> dict[str, str]:
    """Every ‹M10:key› value. ``stories``: {'wr', 'same_forty', 'gap'} → story dicts of ``signature_stories`` (absent → fallback text)."""
    v: dict[str, str] = {"generated_at": utc_now(), "mart_built_at": str(manifest.get("built_at")), "mart_version": str(manifest.get("mart_version")),
                         "run_mode": str(manifest.get("run_mode"))}
    for key, kind in (("wr", "wr"), ("sf", "same_forty"), ("gap", "gap")):
        s = stories.get(kind)
        if s is None:
            for k in ("title", "summary", "narration", "selection", "checklist", "pair", "largest", "distinguishable", "position", "share", "top"):
                v[f"story_{key}_{k}"] = "not available — the selection rule found no eligible case (see notebook 10B)"
            continue
        v[f"story_{key}_title"] = s["title"]
        v[f"story_{key}_summary"] = s["summary_100"]
        v[f"story_{key}_narration"] = s["narration_30s"]
        v[f"story_{key}_selection"] = str(s["provenance"].get("selection", s["provenance"].get("rule", "")))
        v[f"story_{key}_checklist"] = "\n".join(f"- [ ] {c}" for c in s["checklist"])
        if kind != "gap":
            v[f"story_{key}_pair"] = " and ".join(s["card"]["names"]) + f" ({s['card']['position']})"
            v[f"story_{key}_largest"] = str(s["card"].get("largest") or "no distinguishable feature")
            v[f"story_{key}_distinguishable"] = s["card"]["distinguishable"]
        else:
            v["story_gap_position"] = s["card"]["position"]
            v["story_gap_share"] = f"{100 * float(s['card']['share_not_measured']):.0f} %"
            v["story_gap_top"] = str(s["card"].get("top_unmeasured") or "—")
    rows = [(g["position_group"], "—" if g["share_not_measured"] is None else f"{100 * g['share_not_measured']:.0f} %", f"{g['rate_direct']:.1f}", f"{g['rate_partial']:.1f}",
             f"{g['rate_not_measured']:.1f}", f"{g['n_eligible_plays']:,}", g["min_athletes"]) for g in gap_summary]
    v["gap_table"] = _md_table(["Position", "Exposure index not described by the forty", "Direct", "Partial", "Not measured", "Eligible player-plays",
                                "Fewest athletes in a supported cell"], rows)
    v["gap_focus"] = gap_focus["position_group"] if gap_focus else "none (support rule not met)"
    v["gap_n_not_measured_demands"] = str(len({r["demand_id"] for r in gap_matrix if r["coverage_class"] == "not_measured"}))
    cls = {}
    for r in gap_matrix:
        cls.setdefault(r["demand_id"], set()).add(r["class_label"])
    v["gap_class_table"] = _md_table(["Demand", "Class by position (all positions)"], [(d, " / ".join(sorted(c))) for d, c in cls.items()])
    return v


def render(text: str, values: dict[str, str]) -> tuple[str, list[str], list[str]]:
    """(rendered text, filled keys, missing keys)."""
    keys = TOKEN.findall(text)
    filled = sorted({k for k in keys if k in values})
    missing = sorted({k for k in keys if k not in values})
    return TOKEN.sub(lambda m: values.get(m.group(1), m.group(0)), text), filled, missing


def render_documents(report_dir: Path, values: dict[str, str], names: Sequence[str] = DOCUMENTS, write: bool = True) -> pl.DataFrame:
    """Render each authored document from its template (created from the authored file on the first run)."""
    report_dir = Path(report_dir)
    tdir = report_dir / "templates"
    rows = []
    for name in names:
        src, tpl = report_dir / name, tdir / name
        if not tpl.is_file():
            if not src.is_file():
                rows.append({"document": name, "tokens": 0, "filled": 0, "missing": "document absent", "written": False})
                continue
            if write:
                tdir.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(src, tpl)
        text = (tpl if tpl.is_file() else src).read_text(encoding="utf-8")
        out, filled, missing = render(text, values)
        if write:
            src.write_text(out, encoding="utf-8")
        rows.append({"document": name, "tokens": len(set(TOKEN.findall(text))), "filled": len(filled), "missing": ", ".join(missing), "written": write})
    return pl.DataFrame(rows, schema={"document": pl.String, "tokens": pl.Int64, "filled": pl.Int64, "missing": pl.String, "written": pl.Boolean})


def static_anchors(overview: dict[str, Any], T: dict[str, pl.DataFrame]) -> dict[str, str]:
    """The Milestone 09 numbers typed in the authored documents, recomputed from the mart (formatted as typed)."""
    a = {"prospects": f"{overview['n_prospects']:,}", "tracked prospects": f"{overview['n_tracked_prospects']:,}", "tracked attempts": f"{overview['n_tracked_attempts']:,}",
         "NFL episodes": f"{overview['n_nfl_episodes']:,}", "twin pairs": f"{overview['n_twin_pairs']:,}", "distinguishable pairs": f"{overview['n_twin_pairs_distinguishable']:,}",
         "case studies": str(overview["n_cases"])}
    tw = T.get("stopwatch_twins")
    if tw is not None and tw.height:
        pair = ["Anthony Bradford", "Mason McCormick"]
        h = tw.filter(pl.col("name_a").is_in(pair) & pl.col("name_b").is_in(pair))
        if h.height:
            a["hero forty"] = f"{h['forty_a'][0]:.2f}"
    return a


def check_static_numbers(report_dir: Path, overview: dict[str, Any], T: dict[str, pl.DataFrame], names: Sequence[str] = DOCUMENTS) -> pl.DataFrame:
    """For each document: is every anchor value present where the document cites it? (A missing anchor is reported; the
    document is not edited — the author corrects the typed number.)"""
    anchors = static_anchors(overview, T)
    rows = []
    for name in names:
        p = Path(report_dir) / "templates" / name
        p = p if p.is_file() else Path(report_dir) / name
        text = p.read_text(encoding="utf-8") if p.is_file() else ""
        for k, val in anchors.items():
            rows.append({"document": name, "anchor": k, "mart_value": val, "present_in_document": val in text})
    return pl.DataFrame(rows, schema={"document": pl.String, "anchor": pl.String, "mart_value": pl.String, "present_in_document": pl.Boolean})


def figure_index(figures_dir: Path, rel: Any = str) -> pl.DataFrame:
    rows = []
    for k, title, name in FINAL_FIGURES:
        files = {ext: (Path(figures_dir) / f"{name}.{ext}").is_file() for ext in ("png", "svg", "pdf")}
        rows.append({"n": k, "figure": title, "name": name, "path": rel(Path(figures_dir) / f"{name}.png"), **{f"has_{e}": v for e, v in files.items()}})
    return pl.DataFrame(rows)


__all__ = ["DOCUMENTS", "FINAL_FIGURES", "TOKEN", "VERSION", "check_static_numbers", "figure_index", "render", "render_documents", "static_anchors", "token_values"]
