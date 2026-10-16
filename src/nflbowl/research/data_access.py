"""Data access for the unified notebook: two input modes, validation tables, and the per-stage input gates.

Mode A — portable bundle (``nflbowl.data.loading``): ``bundle.json``, ``folds.parquet``, ``sequence_store/{X,M}.npy``,
``meta.parquet``, ``COMPLETE.json``; optional ``athletes.parquet`` (M1 officials + the 12C frozen target) and
``tcn_representations.parquet``. It supports 12F (training, embeddings, label-free evaluation, the inductive transfer
ladder when ``athletes.parquet`` carries the target) and the bundle part of the EDA — NOTHING ELSE.

Mode B — research warehouse (``nflbdb``): a research root with ``warehouse/gold/advanced_movement/…`` written by
Milestones 00–12. 12.0 and 12A–12E, 12G read it through ``ResearchStore``.

Every stage declares its inputs here (``REQUIREMENTS``). ``check_stage`` re-checks the files at the moment the stage
starts, so outputs written earlier in the same run (12.0 → 12C → 12A …) count.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

NS = "advanced_movement"
# (area, name, kind) — kind: parquet | json
Req = tuple[str, str, str]


def _p(area: str, name: str) -> Req:
    return (area, name, "parquet")


def _j(area: str, name: str) -> Req:
    return (area, name, "json")


FOUNDATION_OUT: tuple[Req, ...] = (_j("advanced_v1", "m12_foundation_manifest"), _p("advanced_v1", "m12_cohort_registry"), _p("advanced_v1", "m12_fold_manifest"))
CURVE_TABLES: tuple[Req, ...] = (_p("advanced_v1", "m12_combine_curves"), _p("advanced_v1", "m12_combine_speed_time"), _p("advanced_v1", "m12_combine_hist"),
                                 _p("advanced_v1", "m12_combine_quality"))


@dataclass(frozen=True)
class StageRequirements:
    stage: str
    mode: str                                   # bundle | warehouse
    required: tuple[Req, ...] = ()
    optional: tuple[Req, ...] = ()
    upstream: tuple[str, ...] = ()
    note: str = ""


REQUIREMENTS: dict[str, StageRequirements] = {
    "12.0": StageRequirements("12.0", "warehouse",
                              required=(_p("discovery_v1", "m11_officials"), _p("discovery_v1", "m11_folds"), _p("discovery_v1", "m11_attempt_index"),
                                        _p("discovery_v1", "m11_envelopes"), _p("discovery_v1", "m11_envelope_hist"),
                                        _p("intelligence_v1", "nfl_movement_demand_episodes")),
                              optional=CURVE_TABLES,
                              note="the Combine curve tables are rebuilt from the Combine frames only when absent (or force_rebuild_curves)"),
    "12C": StageRequirements("12C", "warehouse", required=(*FOUNDATION_OUT, _p("advanced_v1", "m12_game_top_speed"), _p("advanced_v1", "m12_combine_curves"),
                                                           _p("discovery_v1", "m11_officials"), _p("discovery_v1", "m11_envelopes"),
                                                           _p("discovery_v1", "m11_athlete_features"), _p("discovery_v1", "m11_nfl_targets"),
                                                           _p("discovery_v1", "m11_envelope_hist")),
                             optional=(_p("intelligence_v1", "nfl_movement_demand_episodes"), _j("advanced_v1", "m12d_selected_representation")),
                             upstream=("12.0",), note="M6 (multi-view) only when 12D ran before; replication needs the 09A episodes"),
    "12B": StageRequirements("12B", "warehouse", required=(*FOUNDATION_OUT, _p("discovery_v1", "m11_officials"), _p("discovery_v1", "m11_attempt_features"),
                                                           _p("discovery_v1", "m11_nfl_targets"), _p("discovery_v1", "m11_nfl_target_reliability")),
                             optional=(_p("advanced_v1", "m12c_targets"),), upstream=("12.0",), note="adds the 12C frozen target when present"),
    "12A": StageRequirements("12A", "warehouse", required=(*FOUNDATION_OUT, _j("advanced_v1", "m12c_target_freeze"), _p("advanced_v1", "m12_combine_curves"),
                                                           _p("discovery_v1", "m11_officials"), _p("discovery_v1", "m11_athlete_features"),
                                                           _p("discovery_v1", "m11_envelope_hist")),
                             optional=(_p("advanced_v1", "m12_combine_speed_time"), _p("intelligence_v1", "stopwatch_twins")),
                             upstream=("12.0", "12C"), note="hard gate on the 12C target freeze; elastic part needs speed–time curves (+ twins)"),
    "12D": StageRequirements("12D", "warehouse", required=(*FOUNDATION_OUT, _p("discovery_v1", "m11_officials"), _p("discovery_v1", "m11_athlete_features"),
                                                           _p("discovery_v1", "m11_attempt_features"), _p("discovery_v1", "m11_nfl_targets"),
                                                           _p("discovery_v1", "m11_envelopes")),
                             optional=(_p("advanced_v1", "m12c_targets"),), upstream=("12.0",)),
    "12E": StageRequirements("12E", "warehouse", required=(*FOUNDATION_OUT, _p("advanced_v1", "m12_combine_hist"), _p("discovery_v1", "m11_officials"),
                                                           _p("discovery_v1", "m11_athlete_features"), _p("discovery_v1", "m11_nfl_targets"),
                                                           _p("discovery_v1", "m11_envelope_hist")),
                             optional=(_p("advanced_v1", "m12_combine_quality"), _p("advanced_v1", "m12c_targets"), _p("intelligence_v1", "stopwatch_twins")),
                             upstream=("12.0",), note="12C target for the downstream ladder; twins optional"),
    "12G": StageRequirements("12G", "warehouse", required=(*FOUNDATION_OUT, _j("advanced_v1", "m12c_target_freeze"), _p("advanced_v1", "m12c_targets"),
                                                           _p("discovery_v1", "m11_officials"), _p("discovery_v1", "m11_athlete_features")),
                             optional=(_p("advanced_v1", "m12_combine_curves"), _p("advanced_v1", "m12_combine_hist"), _j("advanced_v1", "m12d_selected_representation")),
                             upstream=("12.0", "12C"), note="tracks that did not run appear as execution_pending"),
    "12F": StageRequirements("12F", "bundle", note="bundle.json, folds.parquet, sequence_store/*; athletes.parquet with nfl.q90_shared for the transfer ladder"),
    "eda": StageRequirements("eda", "bundle", note="bundle part needs a bundle; foundation part needs the 12.0 outputs"),
}
PRODUCES: dict[str, tuple[Req, ...]] = {
    "12.0": (*FOUNDATION_OUT, *CURVE_TABLES, _p("advanced_v1", "m12_game_top_speed"), _p("advanced_v1", "m12_nfl_bin_support")),
    "12C": (_j("advanced_v1", "m12c_target_freeze"), _p("advanced_v1", "m12c_targets")),
    "12D": (_j("advanced_v1", "m12d_selected_representation"),),
}


# ============================================================================== warehouse location
def gold_dir(research_root: Path | str, run_mode: str = "full") -> Path:
    """``<warehouse>/gold/advanced_movement[/_sample]`` honouring ``configs/phase01.toml`` / ``NFL_BDB_WAREHOUSE`` of that root."""
    from nflbdb.data.config import Phase01Config

    cfg = Phase01Config.load(project_root=research_root)
    base = cfg.paths.gold / NS
    return base / "_sample" if run_mode == "smoke" else base


def req_path(gold: Path, r: Req) -> Path:
    area, name, kind = r
    return gold / area / f"{name}.{kind}"


def warehouse_present(research_root: Path | str | None, run_mode: str = "full") -> bool:
    if not research_root or not Path(research_root).is_dir():
        return False
    try:
        g = gold_dir(research_root, run_mode)
    except Exception:                                           # noqa: BLE001 — unreadable phase01.toml etc.
        return False
    return (g / "discovery_v1").is_dir()


# ============================================================================== bundle location
@dataclass
class DataSources:
    bundle_dir: Path | None
    bundle_note: str
    research_root: Path | None
    warehouse_run_mode: str
    warehouse: bool
    mode: str                                                   # bundle | warehouse | both | none
    notes: list[str] = field(default_factory=list)

    def table(self) -> Any:
        import polars as pl

        return pl.DataFrame([{"source": "bundle (Mode A)", "available": self.bundle_dir is not None, "location": str(self.bundle_dir or "—"), "note": self.bundle_note},
                             {"source": f"warehouse (Mode B, run mode {self.warehouse_run_mode})", "available": self.warehouse,
                              "location": str(self.research_root or "—"), "note": "; ".join(self.notes)}])


def locate_bundle(bundle_dir: str | None, bundle_zip: str | None, extract_to: str | None = None) -> tuple[Path | None, str]:
    """Bundle directory, extracting a ZIP once when only the archive exists. Tries ``bundle_dir``, a nested bundle one
    level down, then ``bundle_zip`` / ``<bundle_dir>.zip``."""
    if bundle_dir:
        d = Path(bundle_dir)
        if (d / "bundle.json").is_file():
            return d, "bundle.json found"
        nested = sorted(p.parent for p in d.glob("*/bundle.json")) if d.is_dir() else []
        if nested:
            return nested[0], f"bundle found one level down ({nested[0].name})"
    zips = [Path(z) for z in (bundle_zip, f"{bundle_dir}.zip" if bundle_dir else None) if z]
    for z in zips:
        if z.is_file():
            from ..data.builder import extract_bundle_zip

            dest = Path(extract_to) if extract_to else z.parent
            out = extract_bundle_zip(z, dest)
            return out, f"extracted {z.name} → {out}"
    return None, f"no bundle at {bundle_dir!r} (and no ZIP)"


def discover(cfg: Any) -> DataSources:
    """Decide the input mode from what exists. ``cfg``: ``UnifiedConfig``."""
    b, note = (None, "disabled by data_mode='warehouse'") if cfg.data_mode == "warehouse" else locate_bundle(cfg.bundle_dir, cfg.bundle_zip)
    wh = cfg.data_mode != "bundle" and warehouse_present(cfg.research_root, cfg.warehouse_run_mode)
    notes = []
    if cfg.data_mode != "bundle" and not wh:
        notes.append(f"no warehouse under {cfg.research_root!r} (expected warehouse/gold/advanced_movement/discovery_v1)")
    mode = "both" if (b and wh) else "bundle" if b else "warehouse" if wh else "none"
    return DataSources(b, note, Path(cfg.research_root) if wh else None, cfg.warehouse_run_mode, wh, mode, notes)


# ============================================================================== bundle validation
def bundle_checks(b: Any, outer_folds: int = 5) -> tuple[Any, Any]:
    """(checks table, athletes-per-fold × position table) for an opened ``SequenceBundle``. ``open_bundle`` already
    validated shapes, dtypes, meta schema and fold ranges; this adds the content checks a reviewer asks for."""
    import numpy as np
    import polars as pl

    from ..data.validation import CHANNELS, M1_COLS, TRANSFER_TARGET

    rows: list[dict[str, Any]] = []

    def add(name: str, ok: bool | None, detail: str) -> None:
        rows.append({"check": name, "status": "pass" if ok else ("warn" if ok is None else "fail"), "detail": detail})

    info = b.info
    add("manifest.format_version", int(info.get("format_version", 0)) == 1, f"format_version {info.get('format_version')} · {len(info.get('files', {}))} files")
    add("manifest.files_present", True, f"sizes {'and sha256 ' if b.verified else ''}checked by open_bundle (parquet / json always hashed)")
    add("arrays.shape", True, f"X {tuple(b.X.shape)} {b.X.dtype}, M {tuple(b.M.shape)} {b.M.dtype}")
    chans = list(info.get("channels") or [])
    add("arrays.channels", chans == list(CHANNELS) if chans else None, f"bundle channels {chans or 'not recorded'} vs expected {list(CHANNELS)}")
    meta, folds = b.meta, b.folds
    ids = set(meta["nfl_id"].unique().to_list())
    fmap = dict(zip(*folds.select("nfl_id", "fold").to_dict(as_series=False).values()))
    with_fold = {i for i in ids if fmap.get(int(i)) is not None}
    add("athletes.ids_integer", meta["nfl_id"].dtype.is_integer(), f"{len(ids):,} athletes in meta")
    add("folds.one_row_per_athlete", folds["nfl_id"].n_unique() == folds.height, f"{folds.height:,} fold rows")
    fv = folds["fold"].drop_nulls()
    add("folds.range", bool(fv.len()) and int(fv.min()) >= 0 and int(fv.max()) < outer_folds, f"folds {sorted(set(fv.to_list()))}")
    add("folds.coverage", len(with_fold) > 0, f"{len(with_fold):,} of {len(ids):,} athletes carry a manifest fold; the rest use the 12F fallback nfl_id % {outer_folds}")
    dom = meta.group_by("nfl_id").agg(pl.col("domain").n_unique().alias("k"))
    add("athletes.both_domains", None if dom.filter(pl.col("k") == 2).height == 0 else True,
        f"{dom.filter(pl.col('k') == 2).height:,} athletes have Combine AND NFL sequences (cross-domain retrieval needs them)")
    lens = np.zeros(b.n_sequences, np.int64)
    for s in range(0, b.n_sequences, 65_536):
        lens[s: s + 65_536] = np.asarray(b.M[s: s + 65_536]).sum(1)
    add("sequences.non_empty", int((lens == 0).sum()) == 0, f"{int((lens == 0).sum())} sequences with no valid frame")
    if b.athletes is not None:
        a = b.athletes
        add("athletes_table.one_row_per_athlete", a["nfl_id"].n_unique() == a.height, f"{a.height:,} rows")
        missing = [c for c in ("position_group", "linked", *M1_COLS) if c not in a.columns]
        add("athletes_table.columns", not missing, f"missing {missing}" if missing else "position_group, linked, M1 officials present")
        linked = a.filter(pl.col("linked").fill_null(False)) if "linked" in a.columns else a.head(0)
        lk = set(linked["nfl_id"].to_list())
        add("athletes_table.linked_in_meta", lk <= ids, f"{len(lk - ids)} linked athletes without sequences")
        add("athletes_table.linked_have_fold", all(fmap.get(int(i)) is not None for i in lk), f"{sum(fmap.get(int(i)) is None for i in lk)} linked athletes without a manifest fold")
        tgt = TRANSFER_TARGET in a.columns
        add("athletes_table.transfer_target", tgt or None, f"{TRANSFER_TARGET}: {int(a[TRANSFER_TARGET].is_not_null().sum()):,} non-null" if tgt else f"{TRANSFER_TARGET} absent: no transfer ladder")
        if missing == [] and linked.height:
            nulls = {c: int(linked[c].null_count()) for c in M1_COLS}
            add("athletes_table.m1_nulls_linked", None if any(nulls.values()) else True, f"nulls among linked athletes: {nulls} (the ladder drops incomplete rows per model)")
    else:
        add("athletes_table.present", None, "athletes.parquet absent: 12F transfer ladder and bundle EDA of officials unavailable")
    add("tcn_table.present", True if b.tcn is not None else None, "tcn_representations.parquet present (F1 baseline)" if b.tcn is not None else "absent: F1 TCN-B baseline skipped")
    pos = meta.group_by("nfl_id").agg(pl.col("position_group").first())
    ft = (pl.DataFrame({"nfl_id": list(fmap), "fold": list(fmap.values())}).join(pos, on="nfl_id", how="inner")
          .group_by("position_group", "fold").len().rename({"len": "n_athletes"}).sort("position_group", "fold"))
    return pl.DataFrame(rows), ft


# ============================================================================== warehouse inventory and gates
def warehouse_inventory(research_root: Path | str, run_mode: str = "full") -> Any:
    """Every declared stage input: present?, size, Parquet row count (lazy: footer only)."""
    import polars as pl

    g = gold_dir(research_root, run_mode)
    seen: dict[str, dict[str, Any]] = {}
    for st, r in REQUIREMENTS.items():
        for req, role in [(x, "required") for x in r.required] + [(x, "optional") for x in r.optional]:
            p = req_path(g, req)
            k = str(p)
            if k not in seen:
                n = None
                if p.is_file() and p.suffix == ".parquet":
                    try:
                        n = int(pl.scan_parquet(p).select(pl.len()).collect().item())
                    except Exception:                               # noqa: BLE001
                        n = None
                seen[k] = {"area": req[0], "name": f"{req[1]}.{req[2]}", "present": p.is_file(), "mb": round(p.stat().st_size / 2 ** 20, 2) if p.is_file() else None,
                           "rows": n, "used_by": []}
            seen[k]["used_by"].append(f"{st}({role[0]})")
    return pl.DataFrame([{**v, "used_by": ", ".join(v["used_by"])} for v in seen.values()]).sort("area", "name")


def check_stage(stage: str, research_root: Path | str | None, run_mode: str, bundle_ok: bool) -> dict[str, Any]:
    """Inputs of one stage at this moment: {ok, missing_required, missing_optional, reason}."""
    r = REQUIREMENTS[stage]
    if r.mode == "bundle":
        return {"ok": bundle_ok, "missing_required": [] if bundle_ok else ["data bundle"], "missing_optional": [],
                "reason": "" if bundle_ok else "no data bundle (Mode A input)"}
    if not research_root or not warehouse_present(research_root, run_mode):
        return {"ok": False, "missing_required": ["research warehouse"], "missing_optional": [],
                "reason": "needs the research warehouse (Mode B); the portable M12F bundle does not contain these inputs"}
    g = gold_dir(research_root, run_mode)
    mr = [f"{a}/{n}.{k}" for (a, n, k) in r.required if not req_path(g, (a, n, k)).is_file()]
    mo = [f"{a}/{n}.{k}" for (a, n, k) in r.optional if not req_path(g, (a, n, k)).is_file()]
    return {"ok": not mr, "missing_required": mr, "missing_optional": mo,
            "reason": "" if not mr else f"missing {len(mr)} required input(s): {', '.join(mr[:4])}{' …' if len(mr) > 4 else ''}"}


def plan(cfg: Any, src: DataSources) -> Any:
    """Expected decision per stage given current files AND the outputs enabled upstream stages will write."""
    import polars as pl

    rows, will_produce = [], set()
    order = ("eda", "12.0", "12C", "12B", "12A", "12D", "12E", "12F", "12G")
    g = gold_dir(src.research_root, src.warehouse_run_mode) if src.warehouse else None
    for st in order:
        r = REQUIREMENTS[st]
        en = cfg.enabled(st)
        if r.mode == "bundle":
            ok, miss = src.bundle_dir is not None, ([] if src.bundle_dir is not None else ["data bundle"])
            if st == "12F" and not ok and cfg.build_bundle_from_warehouse and src.warehouse:
                ok, miss = True, []
        elif not src.warehouse:
            ok, miss = False, ["research warehouse"]
        else:
            miss = [f"{a}/{n}.{k}" for (a, n, k) in r.required if not req_path(g, (a, n, k)).is_file() and (a, n, k) not in will_produce]
            ok = not miss
        dec = "disabled" if not en else ("run" if ok else "skip")
        if en and ok and st in PRODUCES:
            will_produce.update(PRODUCES[st])
        rows.append({"stage": st, "input_mode": r.mode, "enabled": en, "inputs_ready": ok, "expected": dec,
                     "missing": ", ".join(miss[:5]) + (" …" if len(miss) > 5 else ""), "note": r.note})
    return pl.DataFrame(rows)


def read_json(p: Path) -> dict[str, Any] | None:
    try:
        return json.loads(Path(p).read_text(encoding="utf-8")) if Path(p).is_file() else None
    except (OSError, json.JSONDecodeError):
        return None


__all__ = ["CURVE_TABLES", "DataSources", "FOUNDATION_OUT", "PRODUCES", "REQUIREMENTS", "StageRequirements", "bundle_checks", "check_stage", "discover",
           "gold_dir", "locate_bundle", "plan", "read_json", "req_path", "warehouse_inventory", "warehouse_present"]
