"""Milestone 08B.6 — final evidence registry and competition package (Polars; reads persisted artifacts only).

    CLAIMS                 every claim of the final narrative with the artifact it must be read from and its extractor
    evidence_table         claim × value read from the EXECUTED artifact (absent artifact → pending, never a number)
    render_report          refresh the ``‹Cnn: …›`` tokens of a report in place from the evidence table (idempotent)
    figure_index / story_table       every saved figure, and the figures of the core visual story
    reproducibility_manifest · decision_log · gallery_markdown · executive_summary_markdown
    gate_f                 Gate F — does every claim of the narrative trace to an executed artifact?

Evidence levels are never collapsed into one success label:
    measurement_validated · descriptively_repeatable · incremental_information_demonstrated · exploratory_association ·
    nfl_predictive_improvement_established · unsupported_or_unresolved

A token ``‹C08: …›`` in a report is the value of claim C08. Rendering replaces the text after the colon with the value
read from the artifact, or with ``pending — run <notebook>`` when the artifact does not exist. A report therefore
never shows a number that no executed artifact holds, and every number names its claim.
"""

from __future__ import annotations

import json
import platform
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import polars as pl

from ...data.utils import file_fingerprint, read_json, utc_now
from ..contracts import TableContract

S, I, F, B = pl.String(), pl.Int64(), pl.Float64(), pl.Boolean()
LEVELS = (
    ("measurement_validated", "the quantity is measured repeatably and its relation to official tests is quantified"),
    ("descriptively_repeatable", "a representation or feature is repeatable for an athlete; no claim beyond description"),
    ("incremental_information_demonstrated", "repeatable information remains after official tests are accounted for"),
    ("exploratory_association", "an association seen under exploratory status; a hypothesis for independent data"),
    ("nfl_predictive_improvement_established", "held-out improvement of an NFL outcome prediction beyond official testing, under a confirmed definition"),
    ("unsupported_or_unresolved", "tested and not established, or not testable with the current data and definitions"),
)
LEVEL_IDS = tuple(k for k, _ in LEVELS)
QUESTIONS = {
    1: "What is new about 10 Hz Combine tracking?", 2: "Which features are repeatable?", 3: "Which features are already explained by official testing?",
    4: "What did FPCA reveal?", 5: "What did TCN-B learn?", 6: "Why did the first NFL prediction attempt fail to establish added value?",
    7: "How do context and statistical power affect interpretation?", 8: "Are football outcomes more promising?",
    9: "What did the new football transfer test demonstrate?", 10: "What can scouts use today?", 11: "What cannot yet be claimed?",
    12: "What should be tested with future data?"}
NARRATIVE = ("Raw tracking → Kinematics → Reliability → Stopwatch redundancy → FPCA → TCN fingerprint → NFL movement transfer → "
             "Context diagnosis → Football-outcome feasibility → Football transfer → Scouting decision intelligence")
TOKEN = re.compile(r"‹(C\d{2})(?::[^›]*)?›")
PENDING = "pending"

EVIDENCE = TableContract(
    "final_evidence_registry", "Machine-readable evidence registry: every claim of the final narrative and the executed artifact behind it.",
    {"claim_id": S, "question": I, "question_text": S, "claim": S, "evidence_level": S, "milestone": S, "notebook": S, "value": S, "n": I,
     "artifact": S, "artifact_exists": B, "executed": B, "limitation": S, "in_narrative": B}, key=("claim_id",), area="claim",
    docs={"executed": "the artifact exists AND the value could be read from it", "in_narrative": "only executed claims enter the executive narrative",
          "evidence_level": " · ".join(LEVEL_IDS)})
FIGURES = TableContract(
    "final_figure_index", "Index of every saved figure of the advanced-movement programme.",
    {"figure": S, "area": S, "file": S, "run_mode": S, "saved_at": S, "source": S, "n": S, "formats": S}, key=("area", "figure"), area="figure")
STORY = TableContract(
    "final_visual_story", "Core visual story: each step and the saved figure(s) that carry it.",
    {"step": I, "story": S, "figures": S, "n_figures": I, "available": B}, key=("step",), area="story step")
DECISIONS = TableContract(
    "final_scientific_decision_log", "Scientific decision log (transcribed from docs/RESEARCH_DECISIONS.md plus the 08B registry).",
    {"decision_id": S, "milestone": S, "decision": S, "evidence": S, "limitation": S, "status": S}, key=("decision_id",), area="decision")
GATE_F = TableContract("football_gate_f_package", "Gate F — competition package.", {"criterion": S, "passed": B, "detail": S},
                       key=("criterion",), area="criterion")


# ============================================================================== claims
@dataclass(frozen=True)
class Claim:
    claim_id: str
    question: int
    level: str                                   # default evidence level (an extractor may lower or raise it from the artifact)
    milestone: str
    notebook: str
    artifact: str                                # path relative to the advanced-movement tables root (or data root for .json)
    claim: str
    extract: Callable[[Any], dict[str, Any] | None]
    limitation: str = ""


def _f(x: Any, d: int = 2, sign: bool = False) -> str:
    return "n/a" if x is None else f"{float(x):{'+' if sign else ''}.{d}f}"


def _one(df: pl.DataFrame, **eq: Any) -> dict[str, Any] | None:
    for k, v in eq.items():
        if k not in df.columns:
            return None
        df = df.filter(pl.col(k).cast(pl.String) == str(v))
    return df.row(0, named=True) if df.height else None


def _status_row(fid: str, text: str) -> Callable[[pl.DataFrame], dict[str, Any] | None]:
    def fn(df: pl.DataFrame) -> dict[str, Any] | None:
        r = _one(df, feature_id=fid)
        return r and {"value": text.format(rel=_f(r["reliability"]), lo=_f(r["reliability_lower"]), hi=_f(r["reliability_upper"]),
                                           r2=_f(r["r2_stopwatch"]), grade=r["grade"]), "n": int(r["n_athletes_repeat"])}
    return fn


def _m06(df: pl.DataFrame) -> dict[str, Any] | None:
    r = _one(df, analysis="primary", cohort="primary", target_id="O2.accel_sust05_mean", model="M5", reference="M1", metric="delta_rmse")
    return r and {"value": f"Δ RMSE {_f(r['estimate'], 4, True)} [{_f(r['ci_lower'], 4, True)}, {_f(r['ci_upper'], 4, True)}] "
                           f"({_f(100 * r['relative'], 2, True)} % of the official-model error); interval includes 0",
                  "n": int(r["n_athletes"])}


def _fpca(df: pl.DataFrame) -> dict[str, Any] | None:
    r = _one(df, representation="A_speed_by_distance", component=2)
    return r and {"value": f"two components explain {_f(100 * r['cumulative_mean'], 1)} % of held-out curve variance (mean over {r['n_fits']} fold fits)", "n": None}


def _tcn(df: pl.DataFrame) -> dict[str, Any] | None:
    r = _one(df, method="TCNB_A")
    return r and {"value": f"repeat-attempt retrieval AUC {_f(r['auc'], 3)} [{_f(r['auc_lower'], 3)}, {_f(r['auc_upper'], 3)}]", "n": int(r["n_confirm"])}


def _tcn_matched(df: pl.DataFrame) -> dict[str, Any] | None:
    r = _one(df, method="TCNB_A", subset="role:confirm")
    return r and {"value": f"matched-stopwatch retrieval AUC {_f(r['auc_matched'], 3)} [{_f(r['auc_matched_lower'], 3)}, {_f(r['auc_matched_upper'], 3)}]",
                  "n": int(r["n_athletes"])}


def _context(df: pl.DataFrame) -> dict[str, Any] | None:
    r = _one(df, scope="ALL", component="athlete")
    return r and {"value": f"athlete share of within-position NFL window variance {_f(100 * r['share_within_position'], 1)} % "
                           f"[{_f(100 * r['ci_lower'], 1)}, {_f(100 * r['ci_upper'], 1)}]", "n": int(r["n_athletes"])}


def _mde(df: pl.DataFrame) -> dict[str, Any] | None:
    d = df.filter((pl.col("study") == "power") & (pl.col("mode") == "conditional") & (pl.col("scope") == "ALL"))
    if not d.height:
        return None
    parts = [f"{r['candidate']}: " + (f"planted share {_f(r['mde_effect'])} of residual variance (Δ RMSE {_f(100 * r['mde_relative'], 1)} %)"
                                      if r["mde_effect"] is not None else f"80 % power not reached in the grid (max {_f(r['max_power_reached'])})")
             for r in d.iter_rows(named=True)]
    return {"value": "smallest effect detected with 80 % power — " + "; ".join(parts), "n": int(d["n_athletes"][0])}


def _screen(df: pl.DataFrame) -> dict[str, Any] | None:
    r = _one(df, denominator_id="FO01.S0")
    return r and {"value": f"odd / even game reliability {_f(r['rel_sb'])} [{_f(r['rel_lo'])}, {_f(r['rel_hi'])}]; semantic status {r['semantic_status']}; "
                           f"screen status {r['status']}", "n": int(r["n_linked_all_models"])}


def _getoff(df: pl.DataFrame) -> dict[str, Any] | None:
    d = df.filter(pl.col("outcome_id").str.starts_with("FO05"))
    return {"value": f"player_get_off: {d['status'][0]} (source semantics unresolved)", "n": None} if d.height else None


def _registry(doc: dict[str, Any]) -> dict[str, Any] | None:
    if not doc.get("decision"):
        return None
    tg = ", ".join(f"{t['target_id']} ({t['role']}, {t['source_confidence']})" for t in doc.get("targets", [])) or "no target"
    return {"value": f"{doc['decision']} → {doc['approved_status']}: {tg}", "n": None,
            "level": "unsupported_or_unresolved" if doc["approved_status"] in ("stopped", "provisional_review") else "exploratory_association"
            if doc["approved_status"] != "frozen_confirmed" else "measurement_validated"}


def _gate_b(df: pl.DataFrame) -> dict[str, Any] | None:
    d = df.filter((pl.col("role") == "primary") & (pl.col("criterion") == "player_level_stability"))
    return {"value": f"{d['target_id'][0]}: {d['value'][0]} — {'met' if d['passed'][0] else 'NOT met'}", "n": None,
            "level": "measurement_validated" if d["passed"][0] else "unsupported_or_unresolved"} if d.height else None


def _hyp(role: str) -> Callable[[pl.DataFrame], dict[str, Any] | None]:
    def fn(df: pl.DataFrame) -> dict[str, Any] | None:
        d = df.filter(pl.col("role") == role)
        if not d.height:
            return None
        r = d.row(0, named=True)
        txt = (f"{r['model']} versus {r['reference']} on {r['target_id']}: Δ RMSE {_f(r['delta_rmse'], 4, True)} [{_f(r['ci_lower'], 4, True)}, "
               f"{_f(r['ci_upper'], 4, True)}], linkage permutation p {_f(r['p_perm_linkage'], 3)} — {r['reading']} ({r['evidence_label']})")
        level = ("nfl_predictive_improvement_established" if "held-out improvement" in r["reading"] else
                 "exploratory_association" if r["established"] else "unsupported_or_unresolved")
        return {"value": txt, "n": int(r["n_athletes"]), "level": level}
    return fn


def _info(df: pl.DataFrame) -> dict[str, Any] | None:
    if not df.height:
        return None
    good = df.filter(pl.col("status") == "repeatable_beyond_official")["label"].to_list()
    other = [f"{r['label']} ({r['status']})" for r in df.filter(pl.col("status") != "repeatable_beyond_official").iter_rows(named=True)]
    return {"value": "repeatable beyond official tests: " + (", ".join(good) or "none") + ". Others: " + ("; ".join(other) or "none"), "n": int(df["n_athletes"].max()),
            "level": "incremental_information_demonstrated" if good else "descriptively_repeatable"}


def _gate(df: pl.DataFrame) -> dict[str, Any] | None:
    if not df.height:
        return None
    bad = df.filter(~pl.col("passed"))["criterion"].to_list()
    return {"value": "every criterion met" if not bad else "not met: " + ", ".join(bad), "n": None,
            "level": "descriptively_repeatable" if not bad else "unsupported_or_unresolved"}


def _scenarios(df: pl.DataFrame) -> dict[str, Any] | None:
    d = df.filter((pl.col("study") == "sample_size") & (pl.col("scope") == "ALL") & (pl.col("effect") > 0)).sort("effect", "n_athletes")
    if not d.height:
        return None
    top = d.filter(pl.col("effect") == d["effect"].min())
    return {"value": f"power to detect a planted share {_f(top['effect'][0])} of residual variance by cohort size — "
                     + ", ".join(f"n = {r['n_athletes']}: {_f(r['power'])}" for r in top.iter_rows(named=True)), "n": None}


_A, _F8 = "atlas/atlas_feature_status.csv", "football_v1/"
CLAIMS: tuple[Claim, ...] = (
    Claim("C01", 1, "measurement_validated", "04A", "04a_feature_characteristic_atlas", _A,
          "Tracked top speed (p95) of the forty is highly repeatable between attempts",
          _status_row("F2.forty.p95_speed", "ICC {rel} [{lo}, {hi}]; atlas grade {grade}"), "one drill, 10 Hz, eligible recordings only"),
    Claim("C02", 2, "descriptively_repeatable", "04A", "04a_feature_characteristic_atlas", _A,
          "Speed development (time from 4 to 8 yd/s) is repeatable and only partly explained by the stopwatch",
          _status_row("F6.forty.time_v4_to_v8_s", "ICC {rel} [{lo}, {hi}]; R² on the stopwatch {r2}; atlas grade {grade}"),
          "acquisition dependence uncertain; needs the 4 → 8 yd/s interval to be recorded"),
    Claim("C03", 2, "unsupported_or_unresolved", "04A", "04a_feature_characteristic_atlas", _A,
          "Peak tangential acceleration is NOT repeatable enough to describe an athlete",
          _status_row("F2.forty.peak_a_tan", "ICC {rel} [{lo}, {hi}]; atlas grade {grade}"), "single-frame extremes of a 10 Hz derivative"),
    Claim("C04", 3, "measurement_validated", "04A", "04a_feature_characteristic_atlas", _A,
          "Tracked top speed is largely the same information as official testing",
          _status_row("F2.forty.p95_speed", "R² on the stopwatch {r2}"), "within position"),
    Claim("C05", 4, "descriptively_repeatable", "02", "02_functional_pca_baseline", "fpca_explained_variance.csv",
          "The speed-by-distance curve of a forty is low-dimensional", _fpca, "statistical axes, not physical quantities"),
    Claim("C06", 5, "descriptively_repeatable", "04B", "04b_tcn_robustness_and_shortlist", "atlas/robust_method_summary.csv",
          "TCN-B recognises an athlete's repeat attempt: a repeatable movement fingerprint", _tcn, "a similarity space; no dimension has a physical meaning"),
    Claim("C07", 5, "descriptively_repeatable", "04B", "04b_tcn_robustness_and_shortlist", "atlas/robust_matched_summary.csv",
          "The fingerprint still recognises the athlete among prospects with near-identical official results", _tcn_matched,
          "repeatable is not predictive"),
    Claim("C08", 6, "unsupported_or_unresolved", "06B", "06b_combine_to_nfl_transfer_benchmark", "transfer/transfer_paired_comparisons.csv",
          "TCN-B did NOT establish an improvement of held-out NFL sustained-acceleration prediction beyond official testing", _m06,
          "284 athletes; small effects are not detectable (C10); frozen conclusion of Milestone 06"),
    Claim("C09", 7, "measurement_validated", "07A", "07a_nfl_outcome_context_decomposition", "context_transfer/nfl_context_variance_decomposition.csv",
          "NFL movement windows are dominated by context and opportunity, not by the athlete", _context, "observed context labels only"),
    Claim("C10", 7, "measurement_validated", "07C", "07c_detectability_and_information_ceiling", "context_transfer/transfer_mde_estimates.csv",
          "Only large incremental effects were detectable in the Milestone 06 design", _mde, "planted-signal simulation on the real cohort structure"),
    Claim("C11", 8, "exploratory_association", "08A.2", "08a2_outcome_semantics_and_opportunity_validation", "scouting_v1/football_outcome_screen.csv",
          "Recorded receiver separation is a reliable athlete-level quantity with usable linked support", _screen,
          "vendor definition inferred; an interaction outcome; about 91 linked WR / TE"),
    Claim("C12", 8, "unsupported_or_unresolved", "08B.0", "08b0_outcome_resolution_and_freeze", "json:football_v1/football_outcome_registry_v1.json",
          "Football-outcome registry decision", _registry, "a decision never raises a source class"),
    Claim("C13", 8, "measurement_validated", "08B.1", "08b1_football_outcome_reliability", _F8 + "football_gate_b_measurement.csv",
          "Gate B of the primary football outcome (game-split reliability and support)", _gate_b, "reliability is not football skill"),
    Claim("C14", 9, "unsupported_or_unresolved", "08B.2", "08b2_football_transfer_benchmark", _F8 + "football_transfer_hypotheses.csv",
          "Primary football transfer comparison (engineered speed development beyond official testing)", _hyp("primary"),
          "one linked cohort of about 90 receivers"),
    Claim("C15", 9, "unsupported_or_unresolved", "08B.2", "08b2_football_transfer_benchmark", _F8 + "football_transfer_hypotheses.csv",
          "Secondary football transfer comparison (TCN-B beyond official testing)", _hyp("secondary"), "secondary: Holm-adjusted with the primary"),
    Claim("C16", 9, "measurement_validated", "08B.3", "08b3_transfer_robustness_and_detectability", _F8 + "transfer_mde_estimates.csv",
          "Detectability of an incremental effect in the football cohort", _mde, "planted-signal simulation on the real football cohort"),
    Claim("C17", 10, "descriptively_repeatable", "08B.4", "08b4_incremental_movement_information", _F8 + "movement_incremental_information.csv",
          "Combine tracking characteristics that stay repeatable after official tests are accounted for", _info,
          "new measured information about the run; no football implication"),
    Claim("C18", 10, "descriptively_repeatable", "08B.5", "08b5_scouting_decision_intelligence", _F8 + "football_gate_e_scorecard.csv",
          "The scouting scorecard is descriptive, uncertainty-aware and abstains where evidence is missing", _gate, "information support, not a proven decision gain"),
    Claim("C19", 11, "unsupported_or_unresolved", "08A", "08a_football_outcome_feasibility", "scouting_v1/football_outcome_feasibility_registry.csv",
          "Get-off cannot be used as an outcome", _getoff, "no authoritative definition in the repository"),
    Claim("C20", 12, "unsupported_or_unresolved", "08B.3", "08b3_transfer_robustness_and_detectability", _F8 + "transfer_power_scenarios.csv",
          "Cohort size a future study needs", _scenarios, "resampled-cohort scenarios: sampling precision only"),
)
# (story, alternatives of lower-case keywords that must ALL appear in a figure name)
STORY_STEPS: tuple[tuple[str, tuple[tuple[str, ...], ...]], ...] = (
    ("Feature reliability versus redundancy", (("am04a", "reliab"), ("am08b4_01",))),
    ("FPCA curve structure", (("fpca", "component"), ("am02", "component"), ("fpca",))),
    ("TCN matched-stopwatch retrieval", (("am04b", "matched"), ("matched", "retrieval"), ("matched", "stopwatch"))),
    ("Original NFL Δ RMSE (Milestone 06)", (("am06b", "delta"), ("am06b", "paired"), ("am06b", "forest"))),
    ("NFL context decomposition", (("am07a", "variance"), ("am07a", "decomposition"))),
    ("Statistical detectability", (("am07c", "power"), ("am08b3_02",))),
    ("Football-outcome availability", (("am08a", "feasib"), ("am08a2_11",), ("am08b1_10",))),
    ("Separation semantics and reliability", (("am08a2_09",), ("am08b1_05",), ("am08b0_01",))),
    ("Football transfer performance", (("am08b2_02",), ("am08b2_01",))),
    ("Matched-prospect comparison", (("am08b2_07",), ("am08b5_02",), ("am07d_02",))),
    ("Scorecard uncertainty", (("am08b5_04",), ("am07d_05",))),
    ("Final scouting intelligence diagram", (("am08b6_02",),)),
)


# ============================================================================== evidence registry
def _read(path: Path, kind: str) -> Any:
    return read_json(path, {}) if kind == "json" else pl.read_csv(path, infer_schema_length=10000)


def evidence_table(tables_root: Path, data_root: Path, rel: Callable[[Path], str] = str, claims: Sequence[Claim] = CLAIMS) -> pl.DataFrame:
    """One row per claim. ``tables_root`` / ``data_root``: the advanced-movement table and dataset directories of THIS run
    mode. A missing artifact, an unreadable artifact or a missing row all give ``executed = False`` and a pending value."""
    rows = []
    for c in claims:
        kind = "json" if c.artifact.startswith("json:") else "csv"
        path = (data_root / c.artifact.removeprefix("json:")) if kind == "json" else tables_root / c.artifact
        out, exists = None, path.is_file()
        if exists:
            try:
                out = c.extract(_read(path, kind))
            except Exception as e:                       # noqa: BLE001 — an unreadable artifact is reported as pending, with the reason
                out = None
                note = f"artifact present but not readable as expected ({type(e).__name__}: {e})"
            else:
                note = "" if out else "artifact present but the expected row is absent"
        else:
            note = f"run notebook {c.notebook}"
        rows.append({"claim_id": c.claim_id, "question": c.question, "question_text": QUESTIONS[c.question], "claim": c.claim,
                     "evidence_level": (out or {}).get("level", c.level) if out else "unsupported_or_unresolved", "milestone": c.milestone,
                     "notebook": c.notebook, "value": out["value"] if out else f"{PENDING} — {note}", "n": (out or {}).get("n"),
                     "artifact": str(rel(path)), "artifact_exists": exists, "executed": bool(out), "limitation": c.limitation, "in_narrative": bool(out)})
    return pl.DataFrame(rows, schema=EVIDENCE.columns)


def render_report(text: str, evidence: pl.DataFrame) -> tuple[str, list[str]]:
    """Refresh every ``‹Cnn: …›`` token from the evidence table. Returns (text, claim ids still pending). Idempotent."""
    val = {r["claim_id"]: (r["value"], r["executed"]) for r in evidence.iter_rows(named=True)}
    pending: list[str] = []

    def sub(m: re.Match) -> str:
        cid = m.group(1)
        value, ok = val.get(cid, (f"{PENDING} — unknown claim", False))
        if not ok:
            pending.append(cid)
        return f"‹{cid}: {value}›"

    return TOKEN.sub(sub, text), sorted(set(pending))


def tokens_in(text: str) -> list[str]:
    return sorted({m.group(1) for m in TOKEN.finditer(text)})


# ============================================================================== figures
def figure_index(figures_root: Path, rel: Callable[[Path], str] = str) -> pl.DataFrame:
    """Every figure with a metadata sidecar under the advanced-movement figure directory of this run mode."""
    rows = []
    for p in sorted(figures_root.rglob("*.meta.json")) if figures_root.is_dir() else []:
        if "_sample" in p.relative_to(figures_root).parts:
            continue
        m = read_json(p, {})
        area = str(p.parent.relative_to(figures_root)).replace("\\", "/")
        rows.append({"figure": str(m.get("figure", p.name.removesuffix(".meta.json"))), "area": "" if area == "." else area,
                     "file": str(rel(p.with_name(p.name.removesuffix(".meta.json") + ".png"))), "run_mode": str(m.get("run_mode", "")),
                     "saved_at": str(m.get("saved_at", "")), "source": str(m.get("source", "")), "n": str(m.get("n", "")),
                     "formats": " · ".join(m.get("formats", []))})
    return pl.DataFrame(rows, schema=FIGURES.columns).unique(subset=["area", "figure"], maintain_order=True)


def story_table(index: pl.DataFrame) -> pl.DataFrame:
    """The core visual story: strong existing figures are REUSED by name; a step without a saved figure is listed as missing."""
    names = index["figure"].to_list()
    rows = []
    for step, (story, alts) in enumerate(STORY_STEPS, 1):
        hit: list[str] = []
        for keys in alts:
            hit = [n for n in names if all(k in n.lower() for k in keys)]
            if hit:
                break
        rows.append({"step": step, "story": story, "figures": " · ".join(hit[:4]), "n_figures": len(hit), "available": bool(hit)})
    return pl.DataFrame(rows, schema=STORY.columns)


# ============================================================================== manifest, decisions, texts
def reproducibility_manifest(run_mode: str, seed: int, files: dict[str, Path], hashes: dict[str, Any], notebooks: Sequence[str],
                             rel: Callable[[Path], str] = str) -> dict[str, Any]:
    from importlib import metadata

    def version(pkg: str) -> str:
        try:
            return metadata.version(pkg)
        except metadata.PackageNotFoundError:
            return "not installed"

    return {"created_at": utc_now(), "run_mode": run_mode, "seed": seed, "python": platform.python_version(), "platform": platform.platform(),
            "packages": {p: version(p) for p in ("polars", "numpy", "matplotlib", "torch", "duckdb", "pyarrow")},
            "registries_and_configurations": hashes, "notebook_order": list(notebooks),
            "artifacts": {k: {"path": str(rel(p)), "fingerprint": file_fingerprint(p) if p.is_file() else "missing"} for k, p in files.items()},
            "rules": ["athlete-grouped folds; all attempts, games and seasons of an athlete in one fold", "preprocessing fitted on training folds only",
                      "no NFL outcome used to build a Combine representation", "smoke and full artifacts kept apart",
                      "executed by the analyst in Jupyter; the code author ran nothing"]}


def decision_log(markdown: str, registry: dict[str, Any] | None) -> pl.DataFrame:
    """Rows ``| D-… |`` of the decision register, plus the 08B registry decision when one is recorded."""
    rows = []
    for line in markdown.splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split(" | ")]
        if line.strip().startswith("| D-") and len(cells) >= 6:
            rows.append({"decision_id": cells[0], "milestone": cells[1], "decision": cells[2], "evidence": cells[3], "limitation": cells[4], "status": cells[5]})
    if registry and registry.get("decision"):                    # the recorded registry replaces a pending D-08B0-1 line of the register
        tg = ", ".join(f"{t['target_id']} ({t['role']})" for t in registry.get("targets", [])) or "no target"
        rows.append({"decision_id": "D-08B0-1", "milestone": "08B.0", "decision": f"{registry['decision']} → {registry['approved_status']}: {tg}. {registry.get('rationale', '')}",
                     "evidence": f"football_outcome_registry_{registry['version']}.json (hash {registry.get('version_hash')})",
                     "limitation": "source classes are never raised by a decision", "status": "recorded"})
    return pl.DataFrame(rows, schema=DECISIONS.columns).unique(subset=["decision_id"], keep="last", maintain_order=True)


def executive_summary_markdown(evidence: pl.DataFrame) -> str:
    """Concise executive summary: the twelve questions answered from EXECUTED claims only; pending ones are named."""
    lines = ["# NFL Big Data Bowl 2027 — executive summary", "", f"_Generated {utc_now()} from persisted artifacts. {NARRATIVE}._", ""]
    for q, text in QUESTIONS.items():
        rows = evidence.filter(pl.col("question") == q)
        lines.append(f"**{q}. {text}**")
        done = rows.filter(pl.col("in_narrative"))
        lines += [f"- {r['claim']}: {r['value']}" + (f" (n = {r['n']})" if r["n"] is not None else "") + f" — _{r['evidence_level']}_ `{r['claim_id']}`"
                  for r in done.iter_rows(named=True)]
        wait = rows.filter(~pl.col("in_narrative"))
        if wait.height:
            lines.append("- _Not yet answered from an executed artifact:_ " + "; ".join(f"`{r['claim_id']}` ({r['notebook']})" for r in wait.iter_rows(named=True)))
        if not rows.height:
            lines.append("- _No registered claim._")
        lines.append("")
    lines += ["## Evidence hierarchy", ""]
    for level, meaning in LEVELS:
        ids = evidence.filter((pl.col("evidence_level") == level) & pl.col("in_narrative"))["claim_id"].to_list()
        lines.append(f"- **{level}** — {meaning}: " + (", ".join(f"`{i}`" for i in ids) or "_none_"))
    return "\n".join(lines)


def evidence_markdown(evidence: pl.DataFrame) -> str:
    lines = ["| Claim | Question | Statement | Value | n | Evidence level | Artifact | Limitation |", "|---|---|---|---|---|---|---|---|"]
    lines += [f"| `{r['claim_id']}` | {r['question']} | {r['claim']} | {r['value']} | {'' if r['n'] is None else r['n']} | {r['evidence_level']} | "
              f"`{r['artifact']}` | {r['limitation']} |" for r in evidence.iter_rows(named=True)]
    return "\n".join(lines)


def gallery_markdown(cases: pl.DataFrame | None, index: pl.DataFrame) -> str:
    """Scorecard example gallery: the saved scorecard and comparison figures, and the pre-draft rule that selected each case."""
    figs = index.filter(pl.col("figure").str.contains("(?i)scorecard|AM08B5|AM07D_0[12]"))
    lines = ["# Scorecard example gallery", "", "Cases are selected by pre-draft criteria fixed in code (`scouting_intelligence.select_cases`); none is "
             "chosen by an NFL label. They illustrate what the scorecard shows; they are not evidence.", ""]
    if cases is not None and cases.height:
        lines += ["| Case | Position | Athlete A | Athlete B | Matched on | Contrast |", "|---|---|---|---|---|---|"]
        lines += [f"| {r['criterion']} | {r['position_group']} | {r['name_a']} | {r['name_b']} | {r['matched_on']} | {r['contrast']} |" for r in cases.iter_rows(named=True)]
    else:
        lines.append("_No comparison case persisted yet (pending — run notebook 08b5_scouting_decision_intelligence)._")
    lines += ["", "## Saved figures", ""] + ([f"- `{r['file']}` — {r['source']}" for r in figs.iter_rows(named=True)] or ["_None saved yet._"])
    return "\n".join(lines)


def gate_f(evidence: pl.DataFrame, pending_tokens: dict[str, list[str]], story: pl.DataFrame) -> pl.DataFrame:
    """Gate F — does every claim trace to an executed artifact? A pending claim is EXCLUDED from the executive narrative
    (it stays visible as pending); the gate passes only when no report still shows a pending token."""
    n_exec, n_all = int(evidence["executed"].sum()), evidence.height
    still = {k: v for k, v in pending_tokens.items() if v}
    rows = [{"criterion": "narrative_uses_executed_claims_only", "passed": bool((evidence["in_narrative"] == evidence["executed"]).all()),
             "detail": f"{n_exec} of {n_all} claims read from an executed artifact"},
            {"criterion": "evidence_levels_kept_apart", "passed": set(evidence["evidence_level"].unique().to_list()) <= set(LEVEL_IDS),
             "detail": ", ".join(f"{k} {v}" for k, v in evidence.filter(pl.col("executed")).group_by("evidence_level").len().sort("evidence_level").iter_rows())},
            {"criterion": "no_pending_token_in_reports", "passed": not still, "detail": "none pending" if not still else "; ".join(f"{k}: {v}" for k, v in still.items())},
            {"criterion": "visual_story_complete (informative)", "passed": True,
             "detail": f"{int(story['available'].sum())} of {story.height} story steps have a saved figure"}]
    return pl.DataFrame(rows, schema=GATE_F.columns)


__all__ = ["CLAIMS", "Claim", "DECISIONS", "EVIDENCE", "FIGURES", "GATE_F", "LEVELS", "LEVEL_IDS", "NARRATIVE", "PENDING", "QUESTIONS", "STORY",
           "STORY_STEPS", "TOKEN", "decision_log", "evidence_markdown", "evidence_table", "executive_summary_markdown", "figure_index",
           "gallery_markdown", "gate_f", "render_report", "reproducibility_manifest", "story_table", "tokens_in"]
