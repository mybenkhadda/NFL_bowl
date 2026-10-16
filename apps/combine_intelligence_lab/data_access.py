"""Data access of the Combine Intelligence Lab: reads the application mart written by notebook 09E — and nothing else.

No warehouse scan, no model, no computation beyond filtering and re-orienting rows for the selected athlete.

    CIL_MART_DIR    environment variable: the mart directory (default: the full-mode mart under the project root)
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import polars as pl
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if SRC.is_dir() and str(SRC) not in sys.path:          # the interpretation engine is shared with the notebooks (standard library only)
    sys.path.insert(0, str(SRC))

MART_VERSION = "v1"
DEFAULT_MART = ROOT / "warehouse" / "gold" / "advanced_movement" / "intelligence_v1" / "app_mart" / MART_VERSION
REQUIRED = ("prospect_profiles", "prospect_metrics", "combine_attempt_curves", "feature_evidence", "stopwatch_twins", "twin_differences",
            "nfl_movement_demands", "nfl_demand_profiles", "movement_similarity", "movement_similarity_matrix", "scouting_cases", "case_views")


def mart_dir() -> Path:
    return Path(os.environ.get("CIL_MART_DIR") or DEFAULT_MART)


@st.cache_data(show_spinner=False)
def _manifest(path: str, mtime: float) -> dict[str, Any]:
    return json.loads((Path(path) / "mart_manifest.json").read_text(encoding="utf-8"))


def manifest() -> dict[str, Any]:
    p = mart_dir() / "mart_manifest.json"
    return _manifest(str(mart_dir()), p.stat().st_mtime) if p.is_file() else {}


@st.cache_data(show_spinner=False)
def _table(path: str, name: str, stamp: str) -> pl.DataFrame:
    p = Path(path) / f"{name}.parquet"
    return pl.read_parquet(p) if p.is_file() else pl.DataFrame()


def table(name: str) -> pl.DataFrame:
    """A mart table (empty frame when the optional file is absent). Cached per mart build."""
    return _table(str(mart_dir()), name, str(manifest().get("built_at", "")))


@st.cache_data(show_spinner=False)
def _evidence(path: str, stamp: str) -> dict[str, Any]:
    p = Path(path) / "scouting_evidence_registry.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else {}


def evidence() -> dict[str, Any]:
    return _evidence(str(mart_dir()), str(manifest().get("built_at", "")))


def require() -> dict[str, Any]:
    """Stop the page with an actionable message when the mart is absent, incomplete or of another version."""
    m = manifest()
    if not m:
        st.error(f"No application mart at `{mart_dir()}`.")
        st.markdown("Run the notebooks `09a` → `09e` in `notebooks/advanced_movement/` (same `RUN_MODE`), or set the environment variable "
                    "`CIL_MART_DIR` to an existing mart directory, then reload this page.")
        st.stop()
    if m.get("mart_version") != MART_VERSION:
        st.error(f"The mart declares version `{m.get('mart_version')}`; this application reads `{MART_VERSION}`. Rerun notebook `09e`.")
        st.stop()
    missing = [k for k in REQUIRED if not (mart_dir() / f"{k}.parquet").is_file()]
    if missing:
        st.error(f"The mart is incomplete (missing {missing}). Rerun notebook `09e_application_data_mart.ipynb`.")
        st.stop()
    return m


# ============================================================================== athletes
def profiles() -> pl.DataFrame:
    return table("prospect_profiles")


def athlete_labels(tracked_only: bool = False, position: str | None = None) -> dict[int, str]:
    p = profiles()
    if tracked_only:
        p = p.filter(pl.col("n_tracked_attempts") > 0)
    if position and position != "All":
        p = p.filter(pl.col("position_group") == position)
    p = p.sort("display_name", "nfl_id", nulls_last=True)
    return {r["nfl_id"]: f"{r['display_name'] or r['nfl_id']} · {r['position_group']} · {r['draft_year'] or 'n/a'}" for r in p.iter_rows(named=True)}


def profile(nfl_id: int) -> dict[str, Any]:
    p = profiles().filter(pl.col("nfl_id") == nfl_id)
    return p.row(0, named=True) if p.height else {}


def name_of(nfl_id: int) -> str:
    r = profile(nfl_id)
    return str(r.get("display_name") or nfl_id)


def metrics(nfl_id: int) -> list[dict[str, Any]]:
    return table("prospect_metrics").filter(pl.col("nfl_id") == nfl_id).sort("feature").to_dicts()


def curves(nfl_id: int) -> list[dict[str, Any]]:
    """Recorded attempts of an athlete: [{attempt, t, dist, speed, accel, interior}] in attempt order."""
    c = table("combine_attempt_curves").filter(pl.col("nfl_id") == nfl_id).sort("attempt_order", "frame")
    out = []
    for (k,), g in c.partition_by(["attempt_order"], as_dict=True, maintain_order=True).items():
        out.append({"attempt": int(k), "t": g["t_s"].to_list(), "dist": g["dist_yd"].to_list(), "speed": g["speed"].to_list(), "accel": g["a_tan"].to_list(),
                    "interior": (~g["boundary"]).to_list()})
    return out


def band(position: str) -> dict[str, list[float]]:
    b = table("position_curve_bands")
    b = b.filter(pl.col("position_group") == position).sort("dist_yd") if b.height else b
    return {k: b[k].to_list() for k in ("dist_yd", "q10", "q50", "q90")} | {"n": int(b["n_runs"][0])} if b.height else {}


def evidence_rows() -> dict[str, dict[str, Any]]:
    return {r["feature"]: r for r in table("feature_evidence").iter_rows(named=True)}


# ============================================================================== twins (oriented: A = the selected athlete)
def twins_of(nfl_id: int) -> list[dict[str, Any]]:
    t = table("stopwatch_twins")
    rows = []
    for r in t.filter((pl.col("nfl_id_a") == nfl_id) | (pl.col("nfl_id_b") == nfl_id)).iter_rows(named=True):
        rows.append(orient_twin(r, nfl_id))
    return sorted(rows, key=lambda r: (-(r["max_abs_z"] or -1.0), r["pair_id"]))


def orient_twin(r: dict[str, Any], nfl_id: int) -> dict[str, Any]:
    """The twin row with the selected athlete as A. Differences keep the convention B − A."""
    if r["nfl_id_a"] == nfl_id:
        return {**r, "flip": False, "other_id": r["nfl_id_b"], "other_name": r["name_b"]}
    sw = {"flip": True, "other_id": r["nfl_id_a"], "other_name": r["name_a"]}
    for k in ("nfl_id", "name", "draft_year", "forty", "ten", "tier"):
        sw[f"{k}_a"], sw[f"{k}_b"] = r[f"{k}_b"], r[f"{k}_a"]
    for k in ("d_forty", "d_ten", "d_weight", "d_height"):
        sw[k] = None if r[k] is None else -r[k]
    return {**r, **sw}


def twin(pair_id: str, nfl_id: int) -> dict[str, Any]:
    t = table("stopwatch_twins").filter(pl.col("pair_id") == pair_id)
    return orient_twin(t.row(0, named=True), nfl_id) if t.height else {}


def differences(pair_id: str, flip: bool) -> list[dict[str, Any]]:
    rows = table("twin_differences").filter(pl.col("pair_id") == pair_id).sort("feature").to_dicts()
    if not flip:
        return rows
    neg = lambda x: None if x is None else -x
    out = []
    for d in rows:
        out.append({**d, "nfl_id_a": d["nfl_id_b"], "nfl_id_b": d["nfl_id_a"], "value_a": d["value_b"], "value_b": d["value_a"], "difference": neg(d["difference"]),
                    "ci_lower": neg(d["ci_upper"]), "ci_upper": neg(d["ci_lower"]), "z": neg(d["z"]), "quicker": {"a": "b", "b": "a"}.get(d["quicker"], "")})
    return out


# ============================================================================== demands and similarity
def demands(position: str) -> pl.DataFrame:
    return table("nfl_movement_demands").filter(pl.col("position_group") == position)


def demand_labels() -> dict[str, str]:
    r = table("nfl_demand_registry")
    return dict(zip(r["demand_id"].to_list(), r["label"].to_list())) if r.height else {}


def similarity(nfl_id: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """(matrix rows, dimension rows) of one athlete."""
    m = table("movement_similarity_matrix").filter(pl.col("nfl_id") == nfl_id).to_dicts()
    s = table("movement_similarity").filter(pl.col("nfl_id") == nfl_id).to_dicts()
    return m, s


def case_views(case_id: str) -> dict[str, dict[str, list[str]]]:
    v = table("case_views").filter(pl.col("case_id") == case_id).sort("view", "section_order", "line_order")
    out: dict[str, dict[str, list[str]]] = {}
    for r in v.iter_rows(named=True):
        out.setdefault(r["view"], {}).setdefault(r["section"], []).append(r["text"])
    return out


# ============================================================================== Milestone 10 additions (same mart, no new table)
STORED_FORTY_CALIPER = 0.03          # the 09C TwinConfig.forty_tol: filters may only NARROW the stored pairs
STORED_TEN_CALIPER = 0.03            # the 09C TwinConfig.ten_tol
TIER_RANK = {"low": 0, "medium": 1, "high": 2}


def _stamp() -> str:
    return str(manifest().get("built_at", ""))


def rows(name: str) -> list[dict[str, Any]]:
    return table(name).to_dicts()


def profiles_by_id() -> dict[int, dict[str, Any]]:
    return {r["nfl_id"]: r for r in profiles().iter_rows(named=True)}


def twins_filtered(nfl_id: int, forty_tol: float = STORED_FORTY_CALIPER, ten_tol: float | None = None, same_class: bool = False,
                   classes: list[int] | None = None, min_tier: str = "any", feature: str | None = None) -> list[dict[str, Any]]:
    """Stored 09C twins of the athlete, NARROWED by the filters (never widened, never re-matched):

    forty_tol ≤ stored caliper · ten_tol: both splits recorded and within it (None = no split condition) · same_class /
    classes: draft cohort of the other athlete · min_tier: both recordings at or above the tier · feature: only pairs
    distinguishable on that feature."""
    forty_tol = min(float(forty_tol), STORED_FORTY_CALIPER)
    out = []
    dist_on = set()
    if feature:
        d = table("twin_differences").filter((pl.col("feature") == feature) & (pl.col("verdict") == "distinguishable"))
        dist_on = set(d["pair_id"].to_list())
    for t in twins_of(nfl_id):
        if t["d_forty"] is None or abs(t["d_forty"]) > forty_tol + 1e-9:
            continue
        if ten_tol is not None and (t.get("d_ten") is None or abs(t["d_ten"]) > min(ten_tol, STORED_TEN_CALIPER) + 1e-9):
            continue
        if same_class and t.get("draft_year_a") != t.get("draft_year_b"):
            continue
        if classes and t.get("draft_year_b") not in classes:
            continue
        if min_tier != "any" and min(TIER_RANK.get(t.get("tier_a") or "", -1), TIER_RANK.get(t.get("tier_b") or "", -1)) < TIER_RANK[min_tier]:
            continue
        if feature and t["pair_id"] not in dist_on:
            continue
        out.append(t)
    return out


def evidence_list() -> list[dict[str, Any]]:
    return rows("feature_evidence")


def claims() -> dict[str, dict[str, Any]]:
    return {c["claim_id"]: c for c in evidence().get("evidence_claims_08b6", [])}


@st.cache_data(show_spinner=False)
def _gap(path: str, stamp: str) -> list[dict[str, Any]]:
    from nflbdb.research.advanced_movement import movement_gap as mg

    return mg.gap_matrix(rows("nfl_demand_registry"), rows("nfl_movement_demands"), rows("movement_matching_summary"))


def gap() -> list[dict[str, Any]]:
    """The 10C movement-gap matrix, derived from three mart tables (cached per mart build)."""
    return _gap(str(mart_dir()), _stamp())


@st.cache_data(show_spinner=False)
def _stories(path: str, stamp: str) -> dict[str, Any]:
    from nflbdb.research.advanced_movement import movement_gap as mg
    from nflbdb.research.advanced_movement import signature_stories as ss

    m = manifest()
    prov = {"mart_version": m.get("mart_version"), "mart_built_at": m.get("built_at"), "run_mode": m.get("run_mode"), "computed_by": "application (same rules as 10B)"}
    prof = profiles_by_id()
    tw = rows("stopwatch_twins")
    by_pair: dict[str, list[dict[str, Any]]] = {}
    for d in rows("twin_differences"):
        by_pair.setdefault(d["pair_id"], []).append(d)
    dem, cl = rows("nfl_movement_demands"), claims()
    out: dict[str, Any] = {"selection": {}}
    cand = ss.wr_candidates(tw, by_pair, prof)
    pick, why = ss.select_wr_pair(cand)
    out["selection"]["wr"] = why
    if pick:
        t = next(x for x in tw if x["pair_id"] == pick["pair_id"])
        out["wr"] = ss.pair_story("wr", t, by_pair.get(t["pair_id"], []), prof, dem, cl, {**prov, "selection": why})
    twins = {x["pair_id"]: x for x in tw}
    case, why2 = ss.select_same_forty(rows("scouting_cases"), twins)
    out["selection"]["same_forty"] = why2
    if case:
        t = twins[case["pair_id"]]
        out["same_forty"] = ss.pair_story("same_forty", t, by_pair.get(t["pair_id"], []), prof, dem, cl, {**prov, "selection": why2, "case_id": case["case_id"]})
    g = gap()
    focus = mg.gap_position(mg.gap_summary(g))
    out["selection"]["gap"] = "largest share of exposure in demands without a forty analogue" if focus else "no position meets the support rule"
    if focus:
        out["gap"] = ss.gap_story(focus, g, cl, prov)
    return out


def stories() -> dict[str, Any]:
    """The three signature stories (rule-selected from the mart, cached per mart build)."""
    return _stories(str(mart_dir()), _stamp())


def diffs_raw(pair_id: str) -> list[dict[str, Any]]:
    return table("twin_differences").filter(pl.col("pair_id") == pair_id).sort("feature").to_dicts()
