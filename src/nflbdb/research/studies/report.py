"""Integrated research report: populate a Markdown template from SAVED study tables only.

Token syntax (in the template)::

    {{tbl:<table name>|<col>=<value>,<col>=<value>|<column>|<format>}}
    {{rows:<table name>}}                      number of rows of a saved table

* ``<table name>`` is any table registered in the studies manifest (``table_index.csv``).
* Filters must select EXACTLY one row; zero rows → ``⟨pending: …⟩``, several → ``⟨ambiguous: …⟩``.
* ``<format>`` is a Python format spec (``.3f``, ``.1%``, ``,``); default ``str``.

Nothing is computed here — the report can only state what executed code saved. Unresolved tokens are
listed at the end of the generated report and returned, so a partially executed workflow produces a
partially filled report that is honest about what is missing.

Command for the USER::

    python -m nflbdb.research.studies --report
"""

from __future__ import annotations

import re
from pathlib import Path

import polars as pl

from .store import StudyStore

TOKEN = re.compile(r"\{\{(tbl|rows):([^}]*)\}\}")
TEMPLATE = Path("reports") / "phase_03_integrated_report_TEMPLATE.md"
OUTPUT = Path("reports") / "phase_03_integrated_report.md"


def _table(store: StudyStore, name: str, cache: dict) -> pl.DataFrame | None:
    if name not in cache:
        entry = store._manifest()["tables"].get(name)
        p = store.cfg.paths.project_root / entry["path"] if entry else None
        cache[name] = pl.read_parquet(p) if p is not None and p.is_file() else None
    return cache[name]


def resolve(kind: str, body: str, store: StudyStore, cache: dict) -> str:
    if kind == "rows":
        t = _table(store, body.strip(), cache)
        return f"{t.height:,}" if t is not None else f"⟨pending: {body}⟩"
    parts = body.split("|")
    if len(parts) < 3:
        return f"⟨bad token: {body}⟩"
    name, filt, col = parts[0].strip(), parts[1].strip(), parts[2].strip()
    fmt = parts[3].strip() if len(parts) > 3 else ""
    t = _table(store, name, cache)
    if t is None:
        return f"⟨pending: {name}⟩"
    if col not in t.columns:
        return f"⟨bad column: {name}.{col}⟩"
    for cond in filter(None, (c.strip() for c in filt.split(","))):
        k, _, v = cond.partition("=")
        if k not in t.columns:
            return f"⟨bad filter column: {name}.{k}⟩"
        t = t.filter(pl.col(k).cast(pl.String) == v)
    if t.height != 1:
        return f"⟨{'pending' if t.height == 0 else 'ambiguous'}: {body}⟩"
    v = t[col][0]
    if v is None:
        return "n/a"
    return format(v, fmt) if fmt else str(v)


def populate(text: str, store: StudyStore) -> tuple[str, list[str]]:
    cache: dict = {}
    unresolved: list[str] = []

    def sub(m: re.Match) -> str:
        out = resolve(m.group(1), m.group(2), store, cache)
        if out.startswith("⟨"):
            unresolved.append(m.group(0))
        return out

    return TOKEN.sub(sub, text), unresolved


def build_report(store: StudyStore, template: Path | None = None, output: Path | None = None) -> list[str]:
    root = store.cfg.paths.project_root
    template, output = root / (template or TEMPLATE), root / (output or OUTPUT)
    text, unresolved = populate(template.read_text(encoding="utf-8"), store)
    text += ("\n\n---\n## Unresolved placeholders\n" + ("\n".join(f"- `{u}`" for u in unresolved)
                                                         if unresolved else "_none_") + "\n")
    output.write_text(text, encoding="utf-8")
    return unresolved
