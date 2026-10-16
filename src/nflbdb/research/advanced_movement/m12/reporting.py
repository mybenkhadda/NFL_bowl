"""Milestone 12 reporting: conclusions generated from executed results, run manifests, and in-place refresh of the
``reports/milestone12`` documents.

Every document carries ONE generated block between ``<!-- M12:RESULTS:START -->`` and ``<!-- M12:RESULTS:END -->``.
Notebooks replace that block from their own persisted tables; the authored text outside it (design, definitions,
limitations) is never touched. No number is ever written by hand: a block without executed artifacts says
``execution_pending``. Smoke runs write to ``reports/milestone12/_sample`` (a copy of the authored document), so a wiring
check can never overwrite a full-run report.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import polars as pl

from ....data.utils import utc_now
from .config import AUTHORED, EXECUTED, INCOMPLETE, INCONCLUSIVE, PENDING, REJECTED, SUPPORTED, TRACKS

START, END = "<!-- M12:RESULTS:START -->", "<!-- M12:RESULTS:END -->"
DOCS = ("RESEARCH_OBJECTIVES.md", "EXPERIMENT_REGISTRY.md", "FUNCTIONAL_TRANSFER.md", "MEASUREMENT_ERROR.md", "DISTRIBUTIONAL_CAPACITY.md",
        "MULTIVIEW_TRAITS.md", "OPTIMAL_TRANSPORT.md", "SHARED_REPRESENTATIONS.md", "ADVANCED_METHOD_COMPARISON.md", "NEGATIVE_RESULTS.md",
        "NEXT_RESEARCH_DIRECTIONS.md", "REPRODUCIBILITY.md")
TRACK_DOC = {"12.0": "EXPERIMENT_REGISTRY.md", "12A": "FUNCTIONAL_TRANSFER.md", "12B": "MEASUREMENT_ERROR.md", "12C": "DISTRIBUTIONAL_CAPACITY.md",
             "12D": "MULTIVIEW_TRAITS.md", "12E": "OPTIMAL_TRANSPORT.md", "12F": "SHARED_REPRESENTATIONS.md", "12G": "ADVANCED_METHOD_COMPARISON.md"}


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
    head = "| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n"
    body = "".join("| " + " | ".join(fmt(r[c], digits) for c in cols) + " |\n" for r in df.head(max_rows).iter_rows(named=True))
    more = f"\n_{df.height - max_rows} more rows in the persisted table_\n" if df.height > max_rows else ""
    return head + body + more


def status_from(verdicts: Sequence[str | None], completed: bool = True) -> str:
    """Track status from its PRIMARY verdicts: supported (all primaries' intervals favour the hypothesis) · rejected (an
    interval reverses it) · inconclusive · incomplete (not every planned unit finished)."""
    if not completed:
        return INCOMPLETE
    v = [x for x in verdicts if x]
    if not v:
        return INCONCLUSIVE
    if any(x in ("model_worse", "reversed") for x in v):
        return REJECTED
    if all(x in ("model_better", "supported") for x in v):
        return SUPPORTED
    return INCONCLUSIVE


def conclusion_markdown(track: str, facts: dict[str, Any]) -> str:
    t = next(x for x in TRACKS if x.track == track)
    L = [f"## {track} — conclusion ({facts.get('status', EXECUTED)}; run mode {facts.get('run_mode', '?')}; generated {utc_now()})", "",
         "### Research question", t.question, "", "### Declared primary hypothesis", t.primary_hypothesis, "",
         "### Primary outcome / comparison", f"{t.primary_outcome} — {t.primary_comparison} (evaluation target: `{t.evaluation_target}`)", "",
         "### Sample", facts.get("sample", "—"), "", "### Main evidence (from persisted tables)"]
    L += [f"- {e}" for e in facts.get("evidence", [])] or ["- none"]
    L += ["", "### Alternative explanations"] + [f"- {e}" for e in facts.get("alternatives", [])]
    L += ["", "### Practical interpretation", facts.get("relevance", "—"), "", "### Limitations"] + [f"- {e}" for e in facts.get("limitations", [])]
    L += ["", "### Next research questions"] + [f"- {e}" for e in facts.get("next", [])]
    return "\n".join(L) + "\n"


def write_conclusion(store: Any, track: str, md: str) -> Path | None:
    if not store.controls.save_tables:
        return None
    d = store.reports_dir / "conclusions"
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{track.replace('.', '_')}_CONCLUSION.md"
    p.write_text(md, encoding="utf-8")
    return p


def run_manifest(store: Any, track: str, provenance: dict[str, Any], outputs: Sequence[str], status: str, extra: dict[str, Any] | None = None) -> None:
    if store.controls.save_tables:
        store.write_json(f"m12_run_{track.replace('.', '_')}", {"track": track, "status": status, "outputs": list(outputs), "finished_at": utc_now(),
                                                                "provenance": provenance, **(extra or {})})


def manifest(store: Any, track: str) -> dict[str, Any] | None:
    p = store.data_dir / f"m12_run_{track.replace('.', '_')}.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else None


def status_table(store: Any) -> pl.DataFrame:
    rows = []
    for t in TRACKS:
        m = manifest(store, t.track)
        rows.append({"track": t.track, "notebook": t.notebook, "status": (m or {}).get("status", PENDING if t.track != "12.0" else PENDING),
                     "finished_at": (m or {}).get("finished_at"), "run_mode": ((m or {}).get("provenance") or {}).get("run_mode"),
                     "config_hash": ((m or {}).get("provenance") or {}).get("config_hash")})
    return pl.DataFrame(rows)


def _doc_path(store: Any, doc: str) -> Path:
    p = store.reports_dir / doc
    if not p.is_file():
        src = store.cfg.paths.project_root / "reports" / "milestone12" / doc
        p.parent.mkdir(parents=True, exist_ok=True)
        if src.is_file() and src != p:
            shutil.copyfile(src, p)
        else:
            p.write_text(f"# {doc}\n\n{START}\n{END}\n", encoding="utf-8")
    return p


def update_doc(store: Any, doc: str, block: str) -> Path | None:
    """Replace the generated block of ``doc`` (authored text outside the markers is kept)."""
    if not store.controls.save_tables:
        return None
    p = _doc_path(store, doc)
    txt = p.read_text(encoding="utf-8")
    new = f"{START}\n_Generated {utc_now()} from executed artifacts (run mode {store.controls.run_mode})._\n\n{block.strip()}\n{END}"
    if START in txt and END in txt:
        a, rest = txt.split(START, 1)
        _, b = rest.split(END, 1)
        txt = a + new + b
    else:
        txt = txt.rstrip() + "\n\n" + new + "\n"
    p.write_text(txt, encoding="utf-8")
    return p


__all__ = ["DOCS", "END", "START", "TRACK_DOC", "ci", "conclusion_markdown", "fmt", "manifest", "md_table", "run_manifest", "status_from", "status_table",
           "update_doc", "write_conclusion"]
