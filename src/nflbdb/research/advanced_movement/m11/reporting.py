"""Milestone 11 conclusions and report rendering — text is assembled ONLY from executed tables.

    conclusion_markdown   the eight sections of brief §20 from a ``facts`` dict built by the notebook from its own tables
    run_manifest          per-notebook manifest (``m11_run_<track>.json``): provenance, outputs, completion status
    render_reports        rewrites the eight documents of ``reports/milestone11`` from whatever artifacts exist; an absent
                          artifact is reported as "authored, not executed" — never filled with a value
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import polars as pl

from ....data.utils import utc_now
from .config import (AUTHORED, COMPLETED_COMPARISON, EXECUTED_EXPLORATORY, INCONCLUSIVE, NOT_FEASIBLE, STUDIES, SUPPORTED_FINDING,
                     UNSUPPORTED_HYPOTHESIS, VERSION)

TRACKS = ("11.0", "11A", "11B", "11C", "11D", "11E", "11F", "11G", "11H", "11I")
SECTIONS = ("Research question", "Hypothesis", "Sample", "Main evidence", "Alternative explanations", "Practical scouting relevance", "Limitations",
            "Next experiment")
DOCS = ("RESEARCH_OBJECTIVES", "EXPERIMENT_REGISTRY", "NOVEL_METRICS", "CROSS_DOMAIN_TRANSFER", "POSITION_SPECIFIC_FINDINGS", "NEGATIVE_RESULTS",
        "NEXT_DISCOVERIES", "REPRODUCIBILITY")


def fmt(x: Any, d: int = 3) -> str:
    if x is None:
        return "—"
    if isinstance(x, (float, np.floating)):
        return "—" if not np.isfinite(x) else f"{x:.{d}f}"
    return str(x)


def ci(est: Any, lo: Any, hi: Any, d: int = 3) -> str:
    return f"{fmt(est, d)} [{fmt(lo, d)}, {fmt(hi, d)}]"


def md_table(df: pl.DataFrame | None, cols: Sequence[str] | None = None, max_rows: int = 40, digits: int = 3) -> str:
    if df is None or df.is_empty():
        return "_no rows_"
    cols = [c for c in (cols or df.columns) if c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in df.head(max_rows).iter_rows(named=True):
        lines.append("| " + " | ".join(fmt(r[c], digits).replace("|", "/") for c in cols) + " |")
    if df.height > max_rows:
        lines.append(f"\n_{df.height - max_rows} more rows in the saved table._")
    return "\n".join(lines)


def comparison_status(verdicts: Sequence[str | None], computed: bool = True) -> str:
    """Status of a track from the verdicts of its PRIMARY comparisons."""
    v = [x for x in verdicts if x]
    if not computed:
        return AUTHORED
    if not v:
        return EXECUTED_EXPLORATORY
    if any(x in ("model_better", "supported", "matched band stronger (interval > 0)") or str(x).startswith("concordant") for x in v):
        return SUPPORTED_FINDING
    if all(x in ("model_worse", "unsupported") or str(x).startswith("discordant") for x in v):
        return UNSUPPORTED_HYPOTHESIS
    return INCONCLUSIVE


def conclusion_markdown(track: str, facts: dict[str, Any]) -> str:
    """``facts``: question, hypothesis, sample (str), evidence (list[str]), alternatives (list[str]), relevance (str),
    limitations (list[str]), next (list[str]), status (one of ``config.RESULT_STATUSES``), run_mode."""
    s = next((x for x in STUDIES if x.track == track), None)
    lines = [f"## {track} — conclusion ({facts.get('status', AUTHORED)}; run mode {facts.get('run_mode', '?')}; generated {utc_now()})", ""]
    blocks = {"Research question": facts.get("question") or (s.question if s else ""), "Hypothesis": facts.get("hypothesis") or (s.hypothesis if s else ""),
              "Sample": facts.get("sample", "—"), "Main evidence": facts.get("evidence", []), "Alternative explanations": facts.get("alternatives", []),
              "Practical scouting relevance": facts.get("relevance", "—"), "Limitations": facts.get("limitations", []), "Next experiment": facts.get("next", [])}
    for name in SECTIONS:
        lines.append(f"### {name}")
        v = blocks[name]
        lines += [f"- {x}" for x in v] if isinstance(v, list) else [str(v)]
        lines.append("")
    if facts.get("run_mode") == "smoke":
        lines.append("> SMOKE RUN — wiring check only. No number above is a result.")
    return "\n".join(lines)


def write_conclusion(store: Any, track: str, md: str) -> Path | None:
    if not store.controls.save_tables:
        return None
    d = store.reports_dir / "conclusions"
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{track.replace('.', '_')}_CONCLUSION.md"
    p.write_text(md + "\n", encoding="utf-8")
    return p


def run_manifest(store: Any, track: str, provenance: dict[str, Any], outputs: Sequence[str], status: str, extra: dict[str, Any] | None = None) -> None:
    if store.controls.save_tables:
        store.write_json(f"m11_run_{track.replace('.', '_')}", {"track": track, "status": status, "outputs": list(outputs), "finished_at": utc_now(),
                                                                "provenance": provenance, **(extra or {})})


# ============================================================================== rendering
def _load(store: Any, name: str) -> pl.DataFrame | None:
    return pl.read_parquet(store.data_dir / f"{name}.parquet") if (store.data_dir / f"{name}.parquet").is_file() else None


def _manifest(store: Any, track: str) -> dict[str, Any] | None:
    p = store.data_dir / f"m11_run_{track.replace('.', '_')}.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else None


def _status_table(store: Any) -> pl.DataFrame:
    rows = []
    for t in TRACKS:
        m = _manifest(store, t)
        rows.append({"track": t, "status": m["status"] if m else AUTHORED, "run_mode": (m or {}).get("provenance", {}).get("run_mode", "—"),
                     "config_hash": (m or {}).get("provenance", {}).get("config_hash", "—"), "finished_at": (m or {}).get("finished_at", "—")})
    return pl.DataFrame(rows)


def render_reports(store: Any) -> dict[str, Path]:
    """Rewrite the eight documents from the artifacts present in THIS run mode. Returns {doc: path}."""
    R = store.reports_dir
    R.mkdir(parents=True, exist_ok=True)
    st = _status_table(store)
    head = lambda title: [f"# {title}", "", f"_Rendered {utc_now()} by `m11.reporting.render_reports` (engine {VERSION}, run mode `{store.controls.run_mode}`) "
                                           "from saved artifacts only. A track marked `authored_not_executed` has produced no number._", ""]
    docs: dict[str, list[str]] = {}
    docs["RESEARCH_OBJECTIVES"] = head("Milestone 11 — research objectives") + [
        "**Question.** Which physical movement characteristics observed at the NFL Combine transfer to NFL game movement, under which conditions, "
        "for which positions, and with what incremental value beyond standardized testing?", "",
        "Five dimensions, never combined into one score: physical comparability · repeatable measurement · cross-domain athlete-level transfer · "
        "position-specific football relevance · practical information value.", "", "This milestone can establish descriptive associations and predictive "
        "transfer (held-out athletes). It cannot establish a causal performance benefit.", "", "## Execution status", "", md_table(st)]
    reg = pl.DataFrame([{"track": s.track, "study": s.study_id, "hypothesis": s.hypothesis, "baseline": s.baseline, "metric": s.metric,
                         "controls": s.negative_controls, "budget": s.budget} for s in STUDIES])
    docs["EXPERIMENT_REGISTRY"] = head("Milestone 11 — experiment registry") + [md_table(reg, max_rows=20), "", "## Status", "", md_table(st)]
    tour = _load(store, "m11_metric_tournament")
    docs["NOVEL_METRICS"] = head("Milestone 11 — novel metrics") + (
        [md_table(tour.filter(pl.col("position_group") == "ALL"), ["metric_id", "family", "unit", "n_athletes", "L1_measurable", "L2_repeatable", "L3_nonredundant",
                                                                   "L4_physically_transferable", "L5_football_outcome_associated", "L6_predictively_validated",
                                                                   "evidence_tier"])] if tour is not None else
        ["_11I not executed: the candidate registry is in `m11.tournament.CANDIDATES`; no level has been evaluated._"])
    mat, b1, c1, e1 = (_load(store, n) for n in ("m11_transfer_matrix", "m11_speed_state_primary", "m11_transition_transfer", "m11_twin_concordance"))
    docs["CROSS_DOMAIN_TRANSFER"] = head("Milestone 11 — cross-domain transfer") + [
        "## 11A transfer matrix", md_table(mat, ["family", "demand_id", "target", "n_athletes", "m2_delta_rmse", "m2_delta_lo", "m2_delta_hi", "assoc_spearman",
                                                   "matrix_class"]), "",
        "## 11B matched speed band vs broad target (primary)", md_table(b1), "", "## 11C transition metrics ↔ NFL transitions",
        md_table(c1, ["combine", "nfl", "kind", "n_athletes", "spearman", "spearman_lo", "spearman_hi", "partial_spearman", "reading"]), "",
        "## 11E stopwatch-twin concordance", md_table(e1, ["combine", "nfl", "kind", "n_pairs", "concordance", "concordance_lo", "concordance_hi",
                                                            "disjoint_concordance", "reading"])]
    h1, h2, d1 = (_load(store, n) for n in ("m11_action_feasibility", "m11_action_results", "m11_development_mixed"))
    docs["POSITION_SPECIFIC_FINDINGS"] = head("Milestone 11 — position-specific findings") + [
        "## 11H football actions — feasibility", md_table(h1, ["action_id", "semantics", "n_linked", "reliability", "state", "reason"]), "",
        "## 11H held-out results (SUPPORTED actions only)", md_table(h2, ["action_id", "model", "reference", "n_athletes", "delta_rmse", "delta_lo", "delta_hi", "verdict"]),
        "", "## 11D development (mixed model by trait)", md_table(d1)]
    neg = []
    if mat is not None and "matrix_class" in mat.columns:
        neg.append("## 11A cells without established transfer\n\n" + md_table(mat.filter(pl.col("matrix_class").is_in(["no_established_transfer", "insufficient_data"])),
                                                                             ["family", "demand_id", "target", "n_athletes", "m2_delta_rmse", "m2_delta_lo", "m2_delta_hi", "matrix_class"]))
    if tour is not None:
        neg.append("## Metrics failing a level\n\n" + md_table(tour.filter(pl.any_horizontal([pl.col(c) == "fail" for c in tour.columns if c.startswith("L")])),
                                                               ["metric_id", "position_group", "L2_repeatable", "L3_nonredundant", "L4_physically_transferable", "L6_predictively_validated"]))
    nf = [r for r in (h1.filter(pl.col("state") == NOT_FEASIBLE).iter_rows(named=True) if h1 is not None else [])]
    if nf:
        neg.append("## Not feasible with these data\n\n" + "\n".join(f"- {r['action_id']}: {r['reason']}" for r in nf))
    docs["NEGATIVE_RESULTS"] = head("Milestone 11 — negative and inconclusive results") + (neg or ["_No executed table yet._"]) + [
        "", "An interval that includes zero is inconclusive at the cohort size shown — never evidence of no effect."]
    syn = store.data_dir / "m11_synthesis.json"
    s = json.loads(syn.read_text(encoding="utf-8")) if syn.is_file() else None
    docs["NEXT_DISCOVERIES"] = head("Milestone 11 — next discoveries") + (
        [f"## {k.replace('_', ' ')}\n" + ("\n".join(f"- {x}" for x in v) if v else "- none") for k, v in s.get("answers", {}).items()] + ["", "## Proposed next experiments"]
        + [f"- {x}" for x in s.get("next", [])] if s else ["_11I not executed._"])
    shards = sorted(p.parent.name + "/" + p.name for p in (store.shards_dir.glob("*/*/COMPLETE.json") if store.shards_dir.is_dir() else []))
    docs["REPRODUCIBILITY"] = head("Milestone 11 — reproducibility") + [
        "Run order (USER, Jupyter): 11.0 → 11B → 11C → 11A → 11E → 11D → 11G → 11H → 11F → 11I, same `RUN_MODE`; the checks cell of each notebook first.",
        "", "## Runs", md_table(st), "", "## Completed checkpoint sets", "\n".join(f"- {x}" for x in shards) or "- none", "",
        "Every table carries its provenance in `m11_run_<track>.json` (notebook, run mode, seed, config hash, time). Checkpoint shards are immutable "
        "(`m11.checkpoint`); a partial set is never published as complete."]
    out = {}
    for name, lines in docs.items():
        p = R / f"{name}.md"
        p.write_text("\n".join(lines) + "\n", encoding="utf-8")
        out[name] = p
    return out


__all__ = ["DOCS", "SECTIONS", "TRACKS", "ci", "comparison_status", "conclusion_markdown", "fmt", "md_table", "render_reports", "run_manifest", "write_conclusion"]
