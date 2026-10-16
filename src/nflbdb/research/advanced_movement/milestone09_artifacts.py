"""Milestone 09E — the application data mart of the Combine Intelligence Lab.

The interactive application reads ONLY this directory (never the warehouse):

    warehouse/gold/advanced_movement/intelligence_v1/app_mart/<MART_VERSION>/
        <table>.parquet            the tables of ``MART_TABLES`` (compact, typed)
        scouting_evidence_registry.json     executed evidence (08B.4 registry, 08B.6 claims) + interpretation registry
        mart_manifest.json         version, run mode, build time, rows / columns / fingerprint of every file, sources

Smoke mode writes under ``…/_sample/intelligence_v1/app_mart`` (a wiring check; the application labels it).
``validate_mart`` is the acceptance check of the data layer: declared version, required columns, row counts equal to
the manifest, every displayed athlete present in the profile table, every case backed by a twin pair, no grade column.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np
import polars as pl

from ...data.utils import atomic_path, atomic_write_json, file_fingerprint, read_json, utc_now
from ..contracts import TableContract
from .scoring import figures as sfig

MART_VERSION = "v1"
MANIFEST = "mart_manifest.json"
EVIDENCE_JSON = "scouting_evidence_registry.json"
FORBIDDEN_COLUMNS = ("grade", "overall_score", "total_score", "potential", "draft_value", "talent", "hidden_star", "sleeper")
MART_TABLES: dict[str, tuple[str, ...]] = {
    "prospect_profiles": ("nfl_id", "display_name", "position_group", "draft_year", "official_forty", "official_ten", "weight", "height", "evidence_tier",
                          "n_tracked_attempts", "n_twins"),
    "prospect_metrics": ("nfl_id", "feature", "label", "unit", "value", "n_attempts", "standard_error", "pct_position", "state"),
    "combine_attempt_curves": ("seq_id", "nfl_id", "attempt_order", "frame", "t_s", "dist_yd", "speed", "a_tan", "boundary"),
    "position_curve_bands": ("position_group", "dist_yd", "q10", "q50", "q90", "n_runs"),
    "feature_evidence": ("feature", "label", "unit", "phase", "higher_is_quicker", "evidence_status", "football_value"),
    "stopwatch_twins": ("pair_id", "position_group", "nfl_id_a", "nfl_id_b", "d_forty", "n_distinguishable", "max_abs_z", "rank_in_position"),
    "twin_differences": ("pair_id", "feature", "value_a", "value_b", "difference", "se_difference", "verdict", "evidence_status"),
    "twin_summary": ("position_group", "feature", "n_tested", "share_distinguishable"),
    "nfl_movement_demands": ("position_group", "demand_id", "label", "n_episodes", "n_athletes", "episodes_per_100_plays", "supported"),
    "nfl_demand_profiles": ("position_group", "demand_id", "feature", "unit", "q10", "q50", "q90", "n_episodes", "n_athletes", "supported"),
    "nfl_demand_distributions": ("position_group", "demand_id", "feature", "bin_lo", "bin_hi", "n_episodes"),
    "nfl_demand_context": ("context_var", "level", "position_group", "demand_id", "n_episodes"),
    "nfl_context_availability": ("context_var", "source", "share_labelled", "status"),
    "nfl_demand_registry": ("demand_id", "label", "physical_definition", "primary_feature", "combine_analogue", "comparability", "limitation"),
    "nfl_demand_examples": ("example_id", "position_group", "demand_id", "frame", "t_s", "in_episode", "speed", "rel_x", "rel_y"),
    "movement_similarity": ("nfl_id", "demand_id", "dimension", "combine_value", "coverage", "coverage_lo", "coverage_hi", "state", "reason"),
    "movement_similarity_matrix": ("nfl_id", "position_group", "demand_id", "mean_coverage", "state", "reading"),
    "movement_matching_summary": ("position_group", "demand_id", "dimension", "n_prospects", "median_coverage"),
    "shared_feature_dictionary": ("dimension", "combine_definition", "nfl_definition", "comparable", "caveat"),
    "scouting_cases": ("case_id", "case_type", "case_label", "pair_id", "nfl_id_a", "nfl_id_b", "selected_using_nfl_outcome"),
    "case_views": ("case_id", "view", "section", "section_order", "line_order", "text"),
}
OPTIONAL_TABLES = ("nfl_demand_examples",)
ATHLETE_TABLES = ("prospect_metrics", "combine_attempt_curves", "movement_similarity", "movement_similarity_matrix")


_S, _I, _F = pl.String(), pl.Int64(), pl.Float64()
VIEWS = TableContract(
    "scouting_case_views", "Scout, position-coach and general-manager views of every case study, one row per sentence.",
    {"case_id": _S, "view": _S, "section": _S, "section_order": _I, "line_order": _I, "text": _S}, key=("case_id", "view", "section_order", "line_order"),
    area="case × view × sentence")
INTERPRETATION = TableContract(
    "scouting_interpretation_registry", "What every displayed metric measures, means, adds to official testing, and cannot say.",
    {"metric": _S, "what_is_measured": _S, "physical_meaning": _S, "difference_from_official_testing": _S, "reliability": _S, "related_football_movements": _S,
     "what_to_look_at_next": _S, "what_cannot_be_concluded": _S}, key=("metric",), area="metric")
CASE_NFL = TableContract(
    "scouting_case_nfl_observed", "Observed NFL movement of the athletes of the one case type that shows it (read AFTER selection; never a selection input).",
    {"case_id": _S, "nfl_id": _I, "name": _S, "outcome_id": _S, "value": _F, "unit": _S, "n_windows": _I}, key=("case_id", "nfl_id"), area="case × athlete")


def mart_path(store: Any, version: str = MART_VERSION) -> Path:
    return Path(store.mart_dir) / version


# ============================================================================== tables built for the application
def curves_table(frames: pl.DataFrame, windows: pl.DataFrame) -> pl.DataFrame:
    """Recorded 10 Hz frames of every eligible forty attempt, as the application draws them (no resampling)."""
    order = windows.select("seq_id", pl.col("attempt_order").cast(pl.Int64))
    return (frames.join(order, on="seq_id", how="inner")
            .select("seq_id", pl.col("nfl_id").cast(pl.Int64), "attempt_order", pl.col("frame").cast(pl.Int64), pl.col("t_rel").cast(pl.Float32).alias("t_s"),
                    pl.col("dist").cast(pl.Float32).alias("dist_yd"), pl.col("speed").cast(pl.Float32), pl.col("a_tan").cast(pl.Float32), pl.col("boundary"))
            .sort("nfl_id", "attempt_order", "frame"))


def bands_table(frames: pl.DataFrame, windows: pl.DataFrame, min_curves: int = 10) -> pl.DataFrame:
    """10th / 50th / 90th percentile of speed by distance for each position group (the 07D ``curve_band``, reused)."""
    rows = []
    for (pos,), g in sorted(windows.partition_by(["position_group"], as_dict=True).items()):
        b = sfig.curve_band(frames, g["seq_id"].to_list(), min_curves)
        rows += [{"position_group": pos, "dist_yd": float(x), "q10": float(a), "q50": float(m), "q90": float(z), "n_runs": int(b["n"])}
                 for x, a, m, z in zip(b["grid"], b["q10"], b["q50"], b["q90"]) if np.isfinite(m)]
    return pl.DataFrame(rows, schema={"position_group": pl.String, "dist_yd": pl.Float64, "q10": pl.Float64, "q50": pl.Float64, "q90": pl.Float64, "n_runs": pl.Int64})


def profiles_table(profile: pl.DataFrame, prospect: pl.DataFrame, twins: pl.DataFrame) -> pl.DataFrame:
    """The 07D pre-draft profile with the tracked-attempt count and the number of stopwatch twins."""
    n_att = prospect.group_by("nfl_id").agg(pl.col("n_attempts").max().cast(pl.Int64).alias("n_tracked_attempts"))
    both = pl.concat([twins.select(pl.col("nfl_id_a").alias("nfl_id")), twins.select(pl.col("nfl_id_b").alias("nfl_id"))]) if twins.height else pl.DataFrame(schema={"nfl_id": pl.Int64})
    n_tw = both.group_by("nfl_id").agg(pl.len().cast(pl.Int64).alias("n_twins"))
    return (profile.join(n_att, on="nfl_id", how="left").join(n_tw, on="nfl_id", how="left")
            .with_columns(pl.col("n_tracked_attempts").fill_null(0), pl.col("n_twins").fill_null(0)).sort("position_group", "display_name", "nfl_id"))


def demands_table(frequency: pl.DataFrame, registry: pl.DataFrame) -> pl.DataFrame:
    return frequency.join(registry.select("demand_id", "label", "primary_feature", "unit", "combine_analogue"), on="demand_id", how="left")


def evidence_document(scouting_registry: Sequence[dict[str, Any]], claims: Sequence[dict[str, Any]], interpretation: Sequence[dict[str, Any]],
                      curve_evidence: Sequence[dict[str, Any]], sources: dict[str, str]) -> dict[str, Any]:
    """``scouting_evidence_registry.json``: everything the Research Evidence page shows, read from executed artifacts."""
    return {"generated_at": utc_now(), "nfl_predictive_status": "not established (Milestone 06 decision case E; Milestone 08B football transfer)",
            "sources": sources, "scouting_evidence_registry_08b4": list(scouting_registry), "evidence_claims_08b6": list(claims),
            "interpretation_registry": list(interpretation), "combine_curve_feature_evidence_09b": list(curve_evidence),
            "use": ["compare prospects with similar official tests", "locate the movement phase in which they differ", "decide which film and which test to examine next"],
            "do_not_use": ["rank prospects", "grade NFL potential", "claim that a Combine movement difference causes or predicts a football outcome"]}


# ============================================================================== write / read / validate
def write_mart(store: Any, tables: dict[str, pl.DataFrame], evidence: dict[str, Any], provenance: dict[str, Any], version: str = MART_VERSION) -> dict[str, Any]:
    """Write every table and the evidence document atomically, then the manifest (last, so a partial mart has none)."""
    out = mart_path(store, version)
    out.mkdir(parents=True, exist_ok=True)
    missing = [k for k in MART_TABLES if k not in tables and k not in OPTIONAL_TABLES]
    if missing:
        raise KeyError(f"application mart is missing table(s) {missing}")
    files = {}
    for name, df in tables.items():
        need = [c for c in MART_TABLES.get(name, ()) if c not in df.columns]
        if need:
            raise KeyError(f"mart table {name!r} lacks column(s) {need}")
        p = out / f"{name}.parquet"
        with atomic_path(p) as tmp:
            df.write_parquet(tmp)
        files[p.name] = {"rows": df.height, "columns": df.columns, "fingerprint": file_fingerprint(p)}
    atomic_write_json(evidence, out / EVIDENCE_JSON)
    files[EVIDENCE_JSON] = {"rows": None, "columns": sorted(evidence), "fingerprint": file_fingerprint(out / EVIDENCE_JSON)}
    manifest = {"mart_version": version, "run_mode": store.controls.run_mode, "built_at": utc_now(), "files": files, "provenance": provenance,
                "overview": overview(tables)}
    atomic_write_json(manifest, out / MANIFEST)
    return manifest


def load_mart(path: Path) -> tuple[dict[str, pl.DataFrame], dict[str, Any], dict[str, Any]]:
    """(tables, evidence document, manifest) of a mart directory."""
    path = Path(path)
    manifest = read_json(path / MANIFEST, {})
    if not manifest:
        raise FileNotFoundError(f"{path / MANIFEST} is missing: run notebook 09e_application_data_mart.ipynb first")
    tables = {p.stem: pl.read_parquet(p) for p in sorted(path.glob("*.parquet"))}
    return tables, read_json(path / EVIDENCE_JSON, {}), manifest


def overview(tables: dict[str, pl.DataFrame]) -> dict[str, Any]:
    """The counts of the Intelligence Overview page, from the mart tables themselves."""
    get = lambda k: tables.get(k, pl.DataFrame())
    prof, curves, dem, tw, cases, ev = (get(k) for k in ("prospect_profiles", "combine_attempt_curves", "nfl_movement_demands", "stopwatch_twins",
                                                           "scouting_cases", "feature_evidence"))
    all_dem = dem.filter(pl.col("position_group") == "ALL") if dem.height else dem
    return {"n_prospects": prof.height, "n_tracked_prospects": int((prof["n_tracked_attempts"] > 0).sum()) if prof.height else 0,
            "positions": sorted(prof["position_group"].drop_nulls().unique().to_list()) if prof.height else [],
            "n_tracked_attempts": int(curves["seq_id"].n_unique()) if curves.height else 0,
            "n_nfl_episodes": int(all_dem["n_episodes"].sum()) if all_dem.height else 0,
            "n_demands_measurable": int((all_dem["n_episodes"] > 0).sum()) if all_dem.height else 0,
            "n_twin_pairs": tw.height, "n_twin_pairs_distinguishable": int((tw["n_distinguishable"] >= 1).sum()) if tw.height else 0,
            "n_measurements": ev.height, "n_cases": cases.height, "n_case_types": int(cases["case_type"].n_unique()) if cases.height else 0,
            "n_reports": prof.height}


def validate_mart(path: Path, version: str = MART_VERSION) -> pl.DataFrame:
    """Acceptance checks of the data layer. Returns (check, passed, detail); nothing is raised so one table shows all."""
    rows = []

    def add(name: str, ok: bool, detail: str) -> None:
        rows.append({"check": name, "passed": bool(ok), "detail": detail})

    try:
        tables, evidence, manifest = load_mart(path)
    except FileNotFoundError as e:
        add("manifest_present", False, str(e))
        return pl.DataFrame(rows, schema={"check": pl.String, "passed": pl.Boolean, "detail": pl.String})
    add("declared_version", manifest.get("mart_version") == version, f"manifest {manifest.get('mart_version')!r}, expected {version!r}")
    missing = [k for k in MART_TABLES if k not in tables and k not in OPTIONAL_TABLES]
    add("tables_present", not missing, "all present" if not missing else f"missing {missing}")
    lacks = {k: [c for c in cols if c not in tables[k].columns] for k, cols in MART_TABLES.items() if k in tables}
    lacks = {k: v for k, v in lacks.items() if v}
    add("required_columns", not lacks, "all present" if not lacks else str(lacks))
    drift = [k for k, df in tables.items() if manifest.get("files", {}).get(f"{k}.parquet", {}).get("rows") != df.height]
    add("rows_match_manifest", not drift, "row counts equal the manifest" if not drift else f"differs: {drift}")
    stale = [f for f, m in manifest.get("files", {}).items() if not (Path(path) / f).is_file() or file_fingerprint(Path(path) / f) != m.get("fingerprint")]
    add("fingerprints_match_manifest", not stale, "files unchanged since the manifest was written" if not stale else f"changed or absent: {stale}")
    prof = tables.get("prospect_profiles")
    known = set(prof["nfl_id"].to_list()) if prof is not None else set()
    add("profile_key_unique", prof is not None and prof["nfl_id"].n_unique() == prof.height, "one profile row per athlete")
    orphan = {k: n for k in ATHLETE_TABLES if k in tables and (n := len(set(tables[k]["nfl_id"].to_list()) - known))}
    tw = tables.get("stopwatch_twins")
    if tw is not None and tw.height:
        n = len((set(tw["nfl_id_a"].to_list()) | set(tw["nfl_id_b"].to_list())) - known)
        if n:
            orphan["stopwatch_twins"] = n
    add("displayed_athletes_exist_in_source", not orphan, "every athlete of every table is in the profile table" if not orphan else f"unknown athletes: {orphan}")
    cases = tables.get("scouting_cases")
    if cases is not None and tw is not None:
        lost = sorted(set(cases["pair_id"].to_list()) - set(tw["pair_id"].to_list()))
        add("cases_backed_by_twin_pairs", not lost, f"{cases.height} case(s), each a ranked twin pair" if not lost else f"cases without a pair: {lost}")
        add("no_case_selected_on_nfl_outcome", not bool(cases["selected_using_nfl_outcome"].any()) if cases.height else True, "no NFL outcome value entered a selection")
        views = tables.get("case_views")
        if views is not None:
            n_views = views.group_by("case_id").agg(pl.col("view").n_unique().alias("n"))
            short = sorted(set(cases["case_id"].to_list()) - set(n_views.filter(pl.col("n") >= 3)["case_id"].to_list()))
            add("three_views_per_case", not short, "scout, coach and general-manager views for every case" if not short else f"incomplete: {short}")
    bad = {k: v for k, df in tables.items() if (v := [c for c in df.columns if any(m in c.lower() for m in FORBIDDEN_COLUMNS)])}
    add("no_grade_or_ranking_column", not bad, "no grade, potential or overall-score column" if not bad else str(bad))
    mat = tables.get("movement_similarity_matrix")
    if mat is not None and mat.height:
        add("similarity_labelled_descriptive", bool(mat["reading"].str.contains("not a prediction").all()), "every similarity row carries its descriptive reading")
        sim = tables.get("movement_similarity")
        if sim is not None:
            un = sim.filter((pl.col("state") != "supported") & pl.col("coverage_lo").is_not_null()).height
            add("unsupported_comparisons_abstain", un == 0, f"{sim.filter(pl.col('state') != 'supported').height} abstention row(s) carry no interval")
    ev = tables.get("feature_evidence")
    if ev is not None and ev.height:
        add("football_value_not_claimed", set(ev["football_value"].unique().to_list()) == {"not established"}, "every measurement row says football value is not established")
    add("evidence_document_present", bool(evidence.get("scouting_evidence_registry_08b4")) and bool(evidence.get("interpretation_registry")),
        "08B.4 registry and interpretation registry present")
    return pl.DataFrame(rows, schema={"check": pl.String, "passed": pl.Boolean, "detail": pl.String})


def assert_valid(report: pl.DataFrame) -> None:
    bad = report.filter(~pl.col("passed"))
    if bad.height:
        raise AssertionError("application mart validation failed:\n" + "\n".join(f"  {r['check']}: {r['detail']}" for r in bad.iter_rows(named=True)))


__all__ = ["ATHLETE_TABLES", "EVIDENCE_JSON", "FORBIDDEN_COLUMNS", "MANIFEST", "MART_TABLES", "MART_VERSION", "OPTIONAL_TABLES", "assert_valid", "bands_table",
           "curves_table", "demands_table", "evidence_document", "load_mart", "mart_path", "overview", "profiles_table", "validate_mart", "write_mart"]
