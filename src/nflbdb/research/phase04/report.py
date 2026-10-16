"""Populate ``reports/PHASE_04_SCIENTIFIC_RESEARCH_REPORT.md`` from SAVED tables only.

Reuses the Phase 03 token engine (``studies.report``): ``{{tbl:<table>|<col>=<v>,…|<column>|<fmt>}}`` and
``{{rows:<table>}}``. Tables are looked up in the Phase 04 manifest first, then in the Phase 03 studies
manifest, so the report can cite executed Phase 03 evidence (A0, A1/A2, A3) next to Phase 04 results.
Unresolved tokens stay visible as ``⟨pending: …⟩`` and are listed at the end with the study that must
produce them — the report never invents a number.

A Phase 04 study table is used only when that study has a COMPLETED run record: a study writes its tables
one by one and records the run after the last one passed its contract, so tables left by a run that failed
part-way are not cited (they would mix with missing siblings and possibly with an older run).

Command for the USER::

    python -m nflbdb.research.phase04 --report
"""

from __future__ import annotations

from pathlib import Path

from ..studies.report import TOKEN, populate, resolve
from ..studies.store import StudyStore
from . import annotation_design as ad
from . import events as c1
from . import quality_eligibility as q
from . import smoothing as b
from . import validation_registry as v

TEMPLATE = Path("reports") / "PHASE_04_SCIENTIFIC_RESEARCH_REPORT_TEMPLATE.md"
OUTPUT = Path("reports") / "PHASE_04_SCIENTIFIC_RESEARCH_REPORT.md"
GATED = ("Q", "B", "C1", "V")       # studies with a run record; their tables count only after a completed run
PRODUCERS = {**{c.name: k for k, m in (("Q", q), ("B", b), ("C1", c1), ("V", v)) for c in m.CONTRACTS},
             **{c.name: "C2" for c in (ad.TASKS, ad.MATCHES, ad.PERFORMANCE, ad.TAXONOMY, ad.COVERAGE)}}
COMMANDS = {**{k: f"python -m nflbdb.research.phase04 --only {k}" for k in GATED},
            "C2": "python -m nflbdb.research.phase04 --annotation-tasks, label the tasks, then "
                  "--evaluate-annotations reports/annotation/phase_04/annotation_labels.csv"}
REASONS = {"pending": "no row matches the token filter", "ambiguous": "several rows match the token filter",
           "bad column": "requested column not in the table", "bad filter column": "filter column not in the table",
           "bad token": "malformed token"}


class CombinedStore:
    """Read-only view over several study stores' table manifests (first store wins on a name clash).
    Tables of the FIRST store that belong to a ``GATED`` study without a completed run are hidden."""

    def __init__(self, *stores: StudyStore):
        self.stores = stores
        self.cfg = stores[0].cfg

    def _manifest(self) -> dict:
        tables: dict = {}
        for s in reversed(self.stores):
            m = s._manifest()
            gate = s is self.stores[0]
            tables.update({k: e for k, e in m["tables"].items()
                           if not gate or e.get("study") not in GATED or e.get("study") in m["runs"]})
        return {"tables": tables, "figures": {}, "runs": {}}


def source_dependencies(unresolved: list[str], store04: StudyStore, store03: StudyStore) -> list[dict[str, str]]:
    """One row per table behind the unresolved tokens: who produces it, why it did not resolve, what to run."""
    combined = CombinedStore(store04, store03)
    m4, m3 = store04._manifest(), store03._manifest()
    cache: dict = {}
    by_table: dict[str, dict] = {}
    for tok in unresolved:
        kind, body = TOKEN.fullmatch(tok).groups()
        d = by_table.setdefault(body.split("|")[0].strip(), {"n": 0, "why": []})
        d["n"] += 1
        why = resolve(kind, body, combined, cache).strip("⟨⟩").split(":")[0]
        if why not in d["why"]:
            d["why"].append(why)
    rows = []
    for name, d in by_table.items():
        study = PRODUCERS.get(name)
        found = "; ".join(REASONS.get(w, w) for w in d["why"])
        if study in GATED:
            source, command = f"Phase 04 Study {study}", COMMANDS[study]
            state = ("no completed run recorded (never run, or the last run failed a contract)"
                     if study not in m4["runs"] else "study completed but table not registered"
                     if name not in m4["tables"] else f"table present — {found}")
        elif study == "C2":
            source, command = "Phase 04 Study C2 (human annotation)", COMMANDS["C2"]
            state = f"table present — {found}" if name in m4["tables"] else "annotation labels not evaluated yet"
        else:
            e = m3["tables"].get(name)
            source = f"Phase 03 Study {e['study']}" if e else "unknown (no Phase 04 contract, not in the Phase 03 manifest)"
            command = f"python -m nflbdb.research.studies --only {e['study']}" if e else "check the token's table name"
            state = f"table present — {found}" if e else "table not registered"
        rows.append({"table": name, "source": source, "state": state, "command": command, "tokens": str(d["n"])})
    return rows


def unresolved_section(unresolved: list[str], store04: StudyStore, store03: StudyStore) -> str:
    out = "\n\n---\n## Unresolved placeholders\n"
    if not unresolved:
        return out + "_none_\n"
    out += "\n".join(f"- `{u}`" for u in unresolved)
    out += ("\n\n### Source dependencies\n| Table | Produced by | State | Command | Tokens |\n|---|---|---|---|---:|\n"
            + "\n".join(f"| `{r['table']}` | {r['source']} | {r['state']} | `{r['command']}` | {r['tokens']} |"
                        for r in source_dependencies(unresolved, store04, store03)))
    return out + "\n"


def build_phase04_report(store04: StudyStore, store03: StudyStore) -> list[str]:
    root = store04.cfg.paths.project_root
    text, unresolved = populate((root / TEMPLATE).read_text(encoding="utf-8"),
                                CombinedStore(store04, store03))    # type: ignore[arg-type]
    (root / OUTPUT).write_text(text + unresolved_section(unresolved, store04, store03), encoding="utf-8")
    return unresolved
