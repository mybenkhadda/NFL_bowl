"""Study A0 — exact drill-down of failed Phase 01 data-quality checks (Phase 03 · M1.1, Workstream A1).

Question     Which Phase 01 checks failed, what exactly is affected (rows, trajectories, athletes,
             Combine-linked athletes), and is each failure a wrong RULE, a real DATA defect, a
             COVERAGE gap, or a defect already HANDLED downstream?
Population   every row of ``reports/tables/data_quality_results.csv`` whose result is in
             ``IntegrityConfig.results`` (FAIL / WARN / ERROR). Phase 01 ran tracking checks on a 2 %
             Bernoulli sample ("finds issues, cannot prove absence"); this study re-measures each
             failure EXACTLY on the same layer (bronze), so affected counts are complete.
Method       dispatch on the check family (``check_id`` prefix):
               uniq:  duplicate-key rows → affected rows / trajectories / athletes; for tables with
                      ``entity_type`` the same key is re-tested with entity_type added and on PLAYER
                      rows only (M1 evidence: Combine BALL rows carry the athlete's nfl_id).
               range: out-of-range values split by entity_type (athlete rows vs others); for Combine
                      s / a, the persisted M1 frames tell whether the coordinate-DERIVED value is also
                      implausible.
               fk:    orphan child keys; affected child rows; season × season_type breakdown; for
                      game tracking, the share of orphan plays carrying a ball_snap tag.
               trk:   trajectory metrics re-counted exactly over ALL persisted M1 trajectories
                      (same physical thresholds as Phase 01; mapping in TRK_METRICS).
             Classification is a pure function of the measured counts (``classify_*``) so it is
             unit-tested; the evidence string records the numbers it used.
Caveat       Classification labels are DIAGNOSTIC HYPOTHESES for user review, not verdicts. An
             unregistered check family gets ``unreviewed`` with Phase 01's own recommendation.

Outputs (StudyStore, area descriptive): A0_dq_failures, A0_dq_failure_breakdown, A0_dq_failure_examples
(+ figure A0_01_dq_failure_dashboard). The brief's ``reports/tables/m1_1/dq_failures.csv`` is
``reports/tables/phase_03/descriptive/A0_dq_failures.csv`` under the studies convention.

Command for the USER (code authored, NOT executed)::

    python -m nflbdb.research.studies --only A0
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import polars as pl

from ...data.config import Phase01Config, duckdb_connect
from ...data.frames import query_df, query_row
from ...data.sources import SOURCES
from ...data.tables import LayerRefs
from ...data.utils import file_fingerprint, get_logger
from ..artifacts import ArtifactStore
from ..contracts import TableContract
from .config import IntegrityConfig, StudiesConfig
from .store import FigureRecord, StudyStore

log = get_logger(__name__)

STUDY = "A0"
S, I, F, B = pl.String(), pl.Int64(), pl.Float64(), pl.Boolean()
CLASSES = ("rule_error", "data_defect", "data_defect_handled", "coverage_gap", "not_reproduced", "unreviewed",
           "drilldown_error")
# trk:<domain>:<metric> → (M1 trajectory column, note on definitional equivalence)
TRK_METRICS: dict[str, tuple[str, str]] = {
    "n_jumps": ("n_jump_breaks", "same threshold (max_step_yds per nominal interval); M1 starts a new segment"),
    "n_gaps": ("n_gap_breaks", "same threshold (gap_factor × nominal); M1 starts a new segment"),
    "n_dup_ts": ("n_dup_ts_removed", "M1 removes exact duplicate timestamps (first kept)"),
    "n_speed_gt_max": ("n_speed_implausible", "M1 counts SMOOTHED derived speed above max_speed_yds (Phase 01: raw)"),
    "n_accel_gt_max": ("n_accel_implausible", "M1 counts SMOOTHED derived |a| above max_accel_yds2 (Phase 01: raw)"),
}
TRK_DATASETS = {"game": "nfl_trajectory_quality", "combine": "combine_trajectory_quality"}
TRK_KEYS = {"game": ["game_id", "play_id", "nfl_id"], "combine": ["event_id", "nfl_id", "attempt"]}
DERIVED_COUNTERPART = {"s": "speed", "a": "accel_mag"}   # vendor column → M1 derived column
SILVER_RENAMES = {"time": "ts"}                           # silver casts `time` to TIMESTAMP `ts`

# ============================================================================== contracts
DQ_FAILURES = TableContract(
    "A0_dq_failures", "One row per failed / warned Phase 01 check with exact affected counts and a diagnosis.",
    {"check_id": S, "check": S, "source_table": S, "columns": S, "severity": S, "phase01_result": S,
     "phase01_mode": S, "phase01_sampled": B, "phase01_value": F, "phase01_denominator": F,
     "expected_condition": S, "observed_condition": S, "drilldown_layer": S, "drilldown_mode": S,
     "n_affected_keys": I, "n_affected_rows": I, "n_affected_trajectories": I, "n_affected_athletes": I,
     "n_affected_linked_athletes": I, "classification": S, "evidence": S, "recommendation": S, "resolution": S,
     "resolved_downstream": S, "blocking": B, "seconds": F},
    key=("check_id",), area="descriptive",
    docs={"phase01_sampled": "Phase 01 measured this check on a sample (counts were partial)",
          "drilldown_mode": "exact | exact (M1 trajectories) | not_run",
          "n_affected_keys": "duplicate keys (uniq) or orphan child keys (fk); null otherwise",
          "n_affected_rows": "athlete rows (frames for tracking) involved in the failure, exact",
          "n_affected_trajectories": "Combine attempts / NFL player-plays involved (tracking tables only)",
          "n_affected_linked_athletes": "affected athletes with Combine tracking (gold.dim_athlete)",
          "classification": f"one of {CLASSES} — a diagnostic hypothesis for review",
          "resolution": "repair_rule | flag_keep | exclude_from_context_features | none | review",
          "resolved_downstream": "yes / no / unknown — whether silver / M1 already neutralises the failure",
          "blocking": "data_defect at a blocking severity and not resolved downstream"})
BREAKDOWN = TableContract(
    "A0_dq_failure_breakdown", "Affected counts of each failed check by facet (entity_type, season, snap tag …).",
    {"check_id": S, "facet": S, "level": S, "n_keys": I, "n_rows": I, "n_trajectories": I, "n_athletes": I},
    key=("check_id", "facet", "level"), area="descriptive")
EXAMPLES = TableContract(
    "A0_dq_failure_examples", "Deterministic example keys per failed check.",
    {"check_id": S, "rank": I, "example_key": S, "detail": S}, key=("check_id", "rank"), area="descriptive")
CONTRACTS = (DQ_FAILURES, BREAKDOWN, EXAMPLES)


# ============================================================================== classification (pure)
def classify_key_duplicates(dup_rows: int, dup_rows_with_entity: int | None, dup_rows_player: int | None
                            ) -> tuple[str, str, str]:
    """(classification, recommendation, resolution) for a key-uniqueness failure."""
    if dup_rows == 0:
        return ("not_reproduced", "Exact re-measurement finds no duplicate keys: re-run Phase 01 validation; if it "
                "persists the sampled query is wrong.", "review")
    if dup_rows_with_entity == 0:
        return ("rule_error", "Grain is incomplete: duplicates vanish when entity_type joins the key. Amend the check "
                "grain to include entity_type (sources.py grain_candidates already lists it).", "repair_rule")
    if dup_rows_player == 0:
        return ("rule_error", "Duplicates involve non-PLAYER rows only; restrict the check to PLAYER rows "
                "(kinematics already use PLAYER rows only).", "repair_rule")
    return ("data_defect", "PLAYER rows share a full key: inspect conflicting values; silver keeps the lowest "
            "_src_row and quarantines the rest — document the rule and count conflicts.", "flag_keep")


def classify_range(n_out: int, n_out_player: int | None, n_derived_also_out: int | None) -> tuple[str, str, str]:
    if n_out == 0:
        return ("not_reproduced", "No value outside the range on the full table: the sampled count was a tail "
                "fluctuation or the check differs; re-run Phase 01.", "review")
    if n_out_player == 0:
        return ("rule_error", "Only non-athlete rows (e.g. BALL) exceed the range; ball speed is legitimately high. "
                "Restrict the plausibility check to PLAYER rows.", "repair_rule")
    if n_derived_also_out == 0:
        return ("data_defect_handled", "Vendor values on PLAYER frames are implausible but the coordinate-derived M1 "
                "value is not: kinematics derive from x, y; vendor s / a are used only for QC. Exclude these frames "
                "from vendor-agreement statistics.", "flag_keep")
    return ("data_defect", "PLAYER frames are implausible (vendor value; derived value also, or not checked): flag "
            "frames; use robust peaks (sustained 0.5 s) and inspect the trajectories before using peak features.",
            "flag_keep")


def classify_fk(n_orphan_keys: int, child: str, parent: str) -> tuple[str, str, str]:
    if n_orphan_keys == 0:
        return ("not_reproduced", "No orphan keys on the full table; re-run Phase 01.", "review")
    if child == "game_tracking" and parent == "player_play":
        return ("coverage_gap", "Tracked keys without a player_play row have no play context (role, side, result). "
                "Keep them for role-agnostic kinematics; exclude them from context-aware features; never impute.",
                "exclude_from_context_features")
    return ("coverage_gap", "Orphan child keys: keep flagged; never repair by name matching.", "flag_keep")


def classify_trk(total: int) -> tuple[str, str, str]:
    if total == 0:
        return ("not_reproduced", "No occurrence over all persisted M1 trajectories.", "none")
    return ("data_defect_handled", "Tracking glitch handled by M1 segmentation (a new segment starts at the break; "
            "the break is never differentiated). Frames next to a break are filter-boundary frames (Study A3).",
            "flag_keep")


def is_blocking(classification: str, severity: str, resolved: str, cfg: IntegrityConfig) -> bool:
    return classification == "data_defect" and severity in cfg.blocking_severities and resolved != "yes"


def parse_range(threshold: str) -> tuple[float, float] | None:
    m = re.match(r"^\s*\[\s*(-?[\d.]+)\s*,\s*(-?[\d.]+)\s*\]\s*$", threshold or "")
    return (float(m.group(1)), float(m.group(2))) if m else None


def parse_fk(check_id: str) -> tuple[str, list[str], str] | None:
    """'fk:game_tracking:game_id,play_id->player_play' → ('game_tracking', ['game_id', 'play_id'], 'player_play')."""
    m = re.match(r"^fk:([^:]+):([^>]+)->(.+)$", check_id)
    if not m:
        return None
    return m.group(1), [c.strip() for c in m.group(2).rstrip("-").split(",")], m.group(3).strip()


def expected_condition(row: dict[str, Any]) -> str:
    fam = row["check_id"].split(":", 1)[0]
    if fam == "uniq":
        return f"0 duplicate-key rows on ({row['columns']})"
    if fam == "range":
        return f"all {row['columns']} values within {row['threshold']}"
    if fam == "fk":
        return f"every child key {row['columns']} has a parent"
    if fam == "trk":
        thr = str(row.get("threshold") or "")
        return f"{row['metric']} = 0" if thr in ("0", "") else f"no {row['metric']} ({thr})"
    return str(row.get("threshold") or "")


# ============================================================================== drill-down (DuckDB)
@dataclass
class Drill:
    keys: int | None = None
    rows: int | None = None
    trajectories: int | None = None
    athletes: int | None = None
    linked: int | None = None
    classification: str = "unreviewed"
    evidence: str = ""
    recommendation: str = ""
    resolution: str = "review"
    resolved: str = "unknown"
    mode: str = "not_run"


def _int(v: Any) -> int | None:
    return None if v is None else int(v)


class DqDrillDown:
    def __init__(self, cfg: Phase01Config, icfg: IntegrityConfig, con=None):
        self.cfg, self.icfg = cfg, icfg
        self.con = con or duckdb_connect(cfg)
        self.refs = LayerRefs(cfg)
        self.m1 = ArtifactStore(cfg, phase="phase_03")
        self.breakdown: list[dict[str, Any]] = []
        self.examples: list[dict[str, Any]] = []
        self._cols: dict[tuple[str, str], list[str]] = {}

    # ------------------------------------------------------------------ helpers
    def ref(self, table: str, layer: str | None = None) -> str:
        return self.refs.sql(layer or self.icfg.layer, table)

    def columns(self, table: str, layer: str | None = None) -> list[str]:
        k = (layer or self.icfg.layer, table)
        if k not in self._cols:
            self._cols[k] = query_df(self.con, f"SELECT * FROM {self.ref(table, layer)} LIMIT 0").columns
        return self._cols[k]

    @staticmethod
    def _on(a: str, b: str, cols: list[str]) -> str:
        return " AND ".join(f"{a}.{c} IS NOT DISTINCT FROM {b}.{c}" for c in cols)

    @staticmethod
    def _traj_sql(table: str, alias: str = "b") -> str | None:
        tk = SOURCES[table].trajectory_key if table in SOURCES else ()
        return f"count(DISTINCT ({', '.join(f'{alias}.{c}' for c in tk)}))" if tk else None

    def _linked_ids(self) -> str | None:
        if not self.refs.files("gold", "dim_athlete"):
            return None
        return f"(SELECT nfl_id FROM {self.refs.sql('gold', 'dim_athlete')} WHERE has_combine_tracking)"

    def _select_affected(self, table: str) -> str:
        """SELECT list: rows / trajectories / athletes / Combine-linked athletes over rows aliased ``b``."""
        cols = self.columns(table)
        traj, linked = self._traj_sql(table), self._linked_ids()
        has_id = "nfl_id" in cols
        exprs = {"n_rows": "count(*)", "n_traj": traj or "NULL",
                 "n_ath": "count(DISTINCT b.nfl_id)" if has_id else "NULL",
                 "n_linked": (f"count(DISTINCT b.nfl_id) FILTER (WHERE b.nfl_id IN {linked})"
                              if has_id and linked else "NULL")}
        return ", ".join(f"CAST({v} AS BIGINT) AS {n}" for n, v in exprs.items())

    def _player_pred(self, table: str) -> str | None:
        """Rows that are athletes (Combine: entity_type = 'PLAYER'; game tracking: nfl_id present)."""
        cols = self.columns(table)
        if "entity_type" in cols:
            return "b.entity_type = 'PLAYER'"
        if table in SOURCES and SOURCES[table].is_tracking and "nfl_id" in cols:
            return "b.nfl_id IS NOT NULL"
        return None

    def _breakdown_by(self, table: str) -> str | None:
        cols = self.columns(table)
        if "entity_type" in cols:
            return "b.entity_type"
        if table in SOURCES and SOURCES[table].is_tracking and "nfl_id" in cols:
            return "CASE WHEN b.nfl_id IS NULL THEN 'football' ELSE 'athlete' END"
        return None

    def _record_breakdown(self, check_id: str, facet: str, df: pl.DataFrame) -> None:
        for r in df.iter_rows(named=True):
            self.breakdown.append({"check_id": check_id, "facet": facet, "level": str(r["level"]),
                                   "n_keys": _int(r.get("n_keys")), "n_rows": _int(r.get("n_rows")),
                                   "n_trajectories": _int(r.get("n_traj")), "n_athletes": _int(r.get("n_ath"))})

    def _record_examples(self, check_id: str, df: pl.DataFrame, detail_cols: list[str]) -> None:
        key_cols = [c for c in df.columns if c not in detail_cols]
        for k, r in enumerate(df.head(self.icfg.n_examples).iter_rows(named=True), 1):
            self.examples.append({"check_id": check_id, "rank": k,
                                  "example_key": "|".join(str(r[c]) for c in key_cols),
                                  "detail": "; ".join(f"{c}={r[c]}" for c in detail_cols)})

    def _dup_rows(self, ref: str, key_sql: str, where: str = "TRUE") -> int:
        return int(query_row(self.con, f"SELECT CAST(coalesce(sum(n), 0) AS BIGINT) AS v FROM (SELECT {key_sql}, "
                                       f"count(*) AS n FROM {ref} b WHERE {where} GROUP BY ALL HAVING count(*) > 1)")["v"])

    # ------------------------------------------------------------------ uniq
    def key_uniqueness(self, row: dict[str, Any]) -> Drill:
        table, key = row["table"], [c.strip() for c in row["columns"].split(",")]
        k = ", ".join(f"b.{c}" for c in key)
        ref = self.ref(table)
        dup_cte = f"WITH d AS (SELECT {k}, count(*) AS n FROM {ref} b GROUP BY ALL HAVING count(*) > 1)"
        base = query_row(self.con, f"{dup_cte} SELECT CAST(coalesce(sum(n), 0) AS BIGINT) AS dup_rows, "
                                   f"CAST(count(*) AS BIGINT) AS dup_keys FROM d")
        dup_rows, dup_keys = int(base["dup_rows"]), int(base["dup_keys"])
        with_ent = player = None
        cols = self.columns(table)
        if "entity_type" in cols and "entity_type" not in key:
            with_ent = self._dup_rows(ref, f"{k}, b.entity_type")
            player = self._dup_rows(ref, k, "b.entity_type = 'PLAYER'")
        aff: dict[str, Any] = {}
        on = self._on("b", "d", key)
        if dup_rows:
            aff = query_row(self.con, f"{dup_cte} SELECT {self._select_affected(table)} FROM {ref} b SEMI JOIN d ON {on}")
            by = self._breakdown_by(table)
            if by:
                self._record_breakdown(row["check_id"], "row_kind", query_df(self.con, (
                    f"{dup_cte} SELECT {by} AS level, {self._select_affected(table)} FROM {ref} b SEMI JOIN d ON {on} "
                    f"GROUP BY 1 ORDER BY 1")))
            self._record_examples(row["check_id"], query_df(self.con, (
                f"{dup_cte} SELECT {', '.join(key)}, n FROM d ORDER BY {', '.join(key)} LIMIT {self.icfg.n_examples}")),
                ["n"])
        cls, rec, res = classify_key_duplicates(dup_rows, with_ent, player)
        resolved = "unknown"
        if table == "combine_tracking" and self.refs.exists("silver", table):
            skey = [SILVER_RENAMES.get(c, c) for c in key] + ["entity_type"]
            sdup = self._dup_rows(self.refs.sql("silver", table), ", ".join(f"b.{c}" for c in skey))
            resolved = "yes" if sdup == 0 else "no"
        ev = (f"dup_rows={dup_rows:,}, dup_keys={dup_keys:,}"
              + (f", dup_rows_with_entity_type={with_ent:,}, dup_rows_player_only={player:,}"
                 if with_ent is not None else "")
              + (f", silver duplicates on key+entity_type: {'none' if resolved == 'yes' else 'present'}"
                 if resolved != "unknown" else ""))
        return Drill(keys=dup_keys, rows=_int(aff.get("n_rows")), trajectories=_int(aff.get("n_traj")),
                     athletes=_int(aff.get("n_ath")), linked=_int(aff.get("n_linked")), classification=cls,
                     evidence=ev, recommendation=rec, resolution=res, resolved=resolved, mode="exact")

    # ------------------------------------------------------------------ range
    def value_range(self, row: dict[str, Any]) -> Drill:
        table, col = row["table"], row["columns"].strip()
        rng = parse_range(row["threshold"])
        if rng is None:
            return Drill(evidence=f"unparseable threshold {row['threshold']!r}", classification="drilldown_error")
        lo, hi = rng
        ref = self.ref(table)
        out_pred = f"(b.{col} < {lo} OR b.{col} > {hi})"
        tot = query_row(self.con, f"SELECT CAST(count(*) AS BIGINT) AS n, min(b.{col}) AS mn, max(b.{col}) AS mx "
                                  f"FROM {ref} b WHERE {out_pred}")
        n_out = int(tot["n"])
        ppred = self._player_pred(table)
        athlete_pred = f"{out_pred} AND {ppred}" if ppred else out_pred
        aff = query_row(self.con, f"SELECT {self._select_affected(table)}, min(b.{col}) AS mn, max(b.{col}) AS mx "
                                  f"FROM {ref} b WHERE {athlete_pred}") if n_out else {}
        n_player = _int(aff.get("n_rows")) if ppred else None
        by =self._breakdown_by(table)
        if n_out and by:
            self._record_breakdown(row["check_id"], "row_kind", query_df(self.con, (
                f"SELECT {by} AS level, {self._select_affected(table)} FROM {ref} b WHERE {out_pred} "
                f"GROUP BY 1 ORDER BY 1")))
        derived_out = None
        files = self.m1.dataset_files("combine_frame_kinematics")
        if n_out and table == "combine_tracking" and col in DERIVED_COUNTERPART and files:
            dcol = DERIVED_COUNTERPART[col]
            fr = (pl.scan_parquet(files, hive_partitioning=True)
                  .filter((pl.col(col) < lo) | (pl.col(col) > hi))
                  .select(pl.len().alias("n_vendor_out"),
                          ((pl.col(dcol) < lo) | (pl.col(dcol) > hi)).sum().alias("n_derived_out"))
                  .collect().row(0, named=True))
            derived_out = int(fr["n_derived_out"] or 0)
            self._record_breakdown(row["check_id"], "m1_player_frames", pl.DataFrame(
                {"level": ["vendor_out_of_range", f"derived_{dcol}_also_out_of_range"],
                 "n_rows": [int(fr["n_vendor_out"] or 0), derived_out]}))
        if n_out:
            key = [c for c in (SOURCES[table].trajectory_key if table in SOURCES else ()) if c in self.columns(table)]
            extra = [c for c in ("entity_type",) if c in self.columns(table)]
            grp = ", ".join(f"b.{c}" for c in [*key, *extra])
            if grp:
                self._record_examples(row["check_id"], query_df(self.con, (
                    f"SELECT {grp}, max(b.{col}) AS max_value, count(*) AS n FROM {ref} b WHERE {out_pred} "
                    f"GROUP BY ALL ORDER BY max_value DESC LIMIT {self.icfg.n_examples}")), ["max_value", "n"])
        cls, rec, res = classify_range(n_out, n_player if ppred else None, derived_out)
        ev = (f"rows out of range={n_out:,} (min={tot['mn']}, max={tot['mx']})"
              + (f", of which athlete rows={n_player:,}" if ppred and n_player is not None else "")
              + (f", M1 PLAYER frames with derived {DERIVED_COUNTERPART.get(col)} also out={derived_out:,}"
                 if derived_out is not None else ""))
        resolved = "yes" if cls in ("rule_error", "data_defect_handled") else "unknown"
        return Drill(rows=(n_player if ppred else n_out), trajectories=_int(aff.get("n_traj")),
                     athletes=_int(aff.get("n_ath")), linked=_int(aff.get("n_linked")), classification=cls,
                     evidence=ev, recommendation=rec, resolution=res, resolved=resolved, mode="exact")

    # ------------------------------------------------------------------ fk
    def fk_orphans(self, row: dict[str, Any]) -> Drill:
        parsed = parse_fk(row["check_id"])
        if parsed is None:
            return Drill(evidence=f"unparseable fk check id {row['check_id']!r}", classification="drilldown_error")
        child, cc, parent = parsed
        ccs = ", ".join(cc)
        bcc = ", ".join(f"b.{c}" for c in cc)
        ppred = self._player_pred(child) or "TRUE"
        o_cte = (f"WITH o AS (SELECT DISTINCT {bcc} FROM {self.ref(child)} b WHERE {ppred} "
                 f"EXCEPT SELECT DISTINCT {ccs} FROM {self.ref(parent)})")
        n_keys = int(query_row(self.con, f"{o_cte} SELECT CAST(count(*) AS BIGINT) AS v FROM o")["v"])
        aff: dict[str, Any] = {}
        ev = [f"orphan keys={n_keys:,}"]
        if n_keys:
            on = self._on("b", "o", cc)
            aff = query_row(self.con, f"{o_cte} SELECT {self._select_affected(child)} FROM {self.ref(child)} b "
                                      f"SEMI JOIN o ON {on} WHERE {ppred}")
            if "game_id" in cc and self.refs.exists(self.icfg.layer, "games"):
                self._record_breakdown(row["check_id"], "season_season_type", query_df(self.con, (
                    f"{o_cte} SELECT CAST(g.season AS VARCHAR) || ' ' || coalesce(CAST(g.season_type AS VARCHAR), '?') "
                    f"AS level, CAST(count(*) AS BIGINT) AS n_keys FROM o JOIN {self.ref('games')} g "
                    f"ON o.game_id = g.game_id GROUP BY 1 ORDER BY 1")))
            if child == "game_tracking" and "event" in self.columns(child) and cc[:2] == ["game_id", "play_id"]:
                snap = int(query_row(self.con, (
                    f"{o_cte} SELECT CAST(count(DISTINCT (b.game_id, b.play_id)) AS BIGINT) AS v FROM {self.ref(child)} b "
                    f"SEMI JOIN o ON {on} WHERE lower(b.event) = 'ball_snap'"))["v"])
                n_plays = int(query_row(self.con, f"{o_cte} SELECT CAST(count(DISTINCT (o.game_id, o.play_id)) AS BIGINT) "
                                                  f"AS v FROM o")["v"])
                ev.append(f"orphan plays with a ball_snap tag={snap:,} of {n_plays:,}")
                self._record_breakdown(row["check_id"], "ball_snap_tag", pl.DataFrame(
                    {"level": ["with_ball_snap", "without_ball_snap"], "n_keys": [snap, n_plays - snap]}))
            if len(cc) == 3 and cc[:2] == ["game_id", "play_id"]:
                present = int(query_row(self.con, (
                    f"{o_cte} SELECT CAST(count(*) AS BIGINT) AS v FROM o SEMI JOIN "
                    f"(SELECT DISTINCT game_id, play_id FROM {self.ref(parent)}) p "
                    f"ON o.game_id = p.game_id AND o.play_id = p.play_id"))["v"])
                ev.append(f"orphan player keys whose play IS in {parent}={present:,} (athlete missing from a present play)")
            self._record_examples(row["check_id"], query_df(self.con, (
                f"{o_cte} SELECT {ccs}, 1 AS n FROM o ORDER BY {ccs} LIMIT {self.icfg.n_examples}")), ["n"])
        cls, rec, res = classify_fk(n_keys, child, parent)
        return Drill(keys=n_keys, rows=_int(aff.get("n_rows")), trajectories=_int(aff.get("n_traj")),
                     athletes=_int(aff.get("n_ath")), linked=_int(aff.get("n_linked")), classification=cls,
                     evidence="; ".join(ev), recommendation=rec, resolution=res,
                     resolved="no" if n_keys else "yes", mode="exact")

    # ------------------------------------------------------------------ trk
    def trajectory_metric(self, row: dict[str, Any]) -> Drill:
        _, domain, metric = row["check_id"].split(":", 2)
        if metric not in TRK_METRICS or domain not in TRK_DATASETS:
            return Drill(evidence=f"no M1 counterpart for {metric!r} ({domain})", mode="not_run",
                         recommendation=str(row.get("recommended_action") or ""))
        col, note = TRK_METRICS[metric]
        files = self.m1.dataset_files(TRK_DATASETS[domain])
        if not files:
            return Drill(evidence=f"M1 dataset {TRK_DATASETS[domain]} not built", mode="not_run",
                         recommendation=str(row.get("recommended_action") or ""))
        t = pl.scan_parquet(files, hive_partitioning=True)
        r = t.select(pl.col(col).sum().cast(pl.Int64).alias("total"), (pl.col(col) > 0).sum().cast(pl.Int64).alias("n_traj"),
                     pl.col("nfl_id").filter(pl.col(col) > 0).n_unique().cast(pl.Int64).alias("n_ath"),
                     pl.len().cast(pl.Int64).alias("n_all")).collect().row(0, named=True)
        keys = TRK_KEYS[domain]
        ex = (t.filter(pl.col(col) > 0).select(*keys, col)
              .sort([col, *keys], descending=[True] + [False] * len(keys)).head(self.icfg.n_examples).collect())
        self._record_examples(row["check_id"], ex, [col])
        total = int(r["total"] or 0)
        cls, rec, res = classify_trk(total)
        return Drill(rows=total, trajectories=int(r["n_traj"] or 0), athletes=int(r["n_ath"] or 0),
                     classification=cls, evidence=f"M1 {col}: total={total:,}, trajectories={int(r['n_traj'] or 0):,} "
                                                  f"of {int(r['n_all']):,}; {note}",
                     recommendation=rec, resolution=res, resolved="yes" if cls == "data_defect_handled" else "unknown",
                     mode="exact (M1 trajectories)")

    # ------------------------------------------------------------------ dispatch
    def drill(self, row: dict[str, Any]) -> Drill:
        fam = row["check_id"].split(":", 1)[0]
        fn = {"uniq": self.key_uniqueness, "range": self.value_range, "fk": self.fk_orphans,
              "trk": self.trajectory_metric}.get(fam)
        if fn is None:
            return Drill(evidence="no drill-down registered for this check family",
                         recommendation=str(row.get("recommended_action") or ""))
        try:
            return fn(row)
        except Exception as exc:  # noqa: BLE001 — one failing drill-down must not hide the others
            log.warning("[A0] drill-down %s failed: %s", row["check_id"], exc)
            return Drill(classification="drilldown_error", evidence=f"{type(exc).__name__}: {exc}",
                         recommendation=str(row.get("recommended_action") or ""))


# ============================================================================== study
def load_phase01_results(cfg: Phase01Config) -> pl.DataFrame:
    p = cfg.paths.tables / "data_quality_results.csv"
    if not p.is_file():
        raise FileNotFoundError(f"{cfg.paths.rel(p)} missing — run Phase 01 validation first")
    return pl.read_csv(p, infer_schema_length=0)   # all strings; numbers cast explicitly below


def select_failures(dq: pl.DataFrame, icfg: IntegrityConfig) -> pl.DataFrame:
    return dq.filter(pl.col("result").is_in(list(icfg.results))).unique("check_id", keep="first", maintain_order=True)


def _float(v: Any) -> float | None:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if np.isfinite(f) else None


def assemble(failures: pl.DataFrame, drills: list[tuple[Drill, float]], icfg: IntegrityConfig) -> pl.DataFrame:
    rows = []
    for r, (d, secs) in zip(failures.iter_rows(named=True), drills):
        mode = str(r.get("mode") or "")
        layer = icfg.layer if d.mode == "exact" else ("gold.phase_03" if d.mode.startswith("exact (M1") else "")
        rows.append({
            "check_id": r["check_id"], "check": r["check"], "source_table": r["table"], "columns": r["columns"],
            "severity": r["severity"], "phase01_result": r["result"], "phase01_mode": mode,
            "phase01_sampled": mode.startswith("sample"), "phase01_value": _float(r.get("value")),
            "phase01_denominator": _float(r.get("denominator")), "expected_condition": expected_condition(r),
            "observed_condition": r.get("details") or "", "drilldown_layer": layer, "drilldown_mode": d.mode,
            "n_affected_keys": d.keys, "n_affected_rows": d.rows, "n_affected_trajectories": d.trajectories,
            "n_affected_athletes": d.athletes, "n_affected_linked_athletes": d.linked,
            "classification": d.classification, "evidence": d.evidence,
            "recommendation": d.recommendation or str(r.get("recommended_action") or ""), "resolution": d.resolution,
            "resolved_downstream": d.resolved,
            "blocking": is_blocking(d.classification, r["severity"], d.resolved, icfg), "seconds": round(secs, 3)})
    return pl.DataFrame(rows, schema=DQ_FAILURES.columns) if rows else DQ_FAILURES.empty()


class DqFailureStudy:
    STUDY = STUDY

    def __init__(self, cfg: Phase01Config, scfg: StudiesConfig | None = None, store: StudyStore | None = None, **_: Any):
        self.cfg = cfg
        self.scfg = scfg or StudiesConfig()
        self.icfg = self.scfg.integrity
        # an exact study: a --sample-athletes store would mislabel it, so it always writes to the full store
        self.store = store if (store is not None and not store.sampled) else StudyStore(cfg, sampled=False)

    def upstream(self, failures: pl.DataFrame) -> dict[str, Any]:
        refs = LayerRefs(self.cfg)
        tables = sorted(set(failures["table"].to_list()) | {"games", "player_play"})
        files: dict[str, Any] = {t: [list(file_fingerprint(p).values()) for p in refs.files(self.icfg.layer, t)]
                                 for t in tables}
        m1 = ArtifactStore(self.cfg, phase="phase_03")
        files["m1"] = {n: [list(file_fingerprint(p).values()) for p in m1.dataset_files(n)]
                       for n in ("nfl_trajectory_quality", "combine_trajectory_quality", "combine_frame_kinematics")}
        files["dq_results"] = list(file_fingerprint(self.cfg.paths.tables / "data_quality_results.csv").values())
        return files

    def compute(self) -> dict[str, pl.DataFrame]:
        failures = select_failures(load_phase01_results(self.cfg), self.icfg)
        log.info("[studies:A0] %d non-PASS checks to drill down", failures.height)
        dd = DqDrillDown(self.cfg, self.icfg)
        drills = []
        for r in failures.iter_rows(named=True):
            t0 = time.perf_counter()
            d = dd.drill(r)
            drills.append((d, time.perf_counter() - t0))
            log.info("[studies:A0] %s → %s (%s)", r["check_id"], d.classification, d.evidence)
        bd = pl.DataFrame(dd.breakdown, schema=BREAKDOWN.columns) if dd.breakdown else BREAKDOWN.empty()
        ex = pl.DataFrame(dd.examples, schema=EXAMPLES.columns) if dd.examples else EXAMPLES.empty()
        return {DQ_FAILURES.name: assemble(failures, drills, self.icfg),
                BREAKDOWN.name: bd.unique(["check_id", "facet", "level"], keep="first", maintain_order=True),
                EXAMPLES.name: ex}

    def fingerprint(self) -> str:
        failures = select_failures(load_phase01_results(self.cfg), self.icfg)
        return self.store.fingerprint(STUDY, self.scfg.section("integrity"), self.upstream(failures))

    def run(self, force: bool = False, figures: bool = True) -> dict[str, pl.DataFrame]:
        fp = self.fingerprint()
        current, reason = self.store.is_current(STUDY, fp, CONTRACTS)
        if current and not force:
            log.info("[studies:A0] reuse (%s)", reason)
            out = self.load()
        else:
            log.info("[studies:A0] compute (%s%s)", "forced; " if force else "", reason)
            res = self.compute()
            out = {c.name: self.store.write(res[c.name], c, STUDY) for c in CONTRACTS}
            self.store.record_run(STUDY, fp, CONTRACTS)
        if figures:
            self.figures(out)
        return out

    def load(self) -> dict[str, pl.DataFrame]:
        return {c.name: self.store.read(c) for c in CONTRACTS}

    def figures(self, out: dict[str, pl.DataFrame]) -> None:
        for rec, fn in FIGURES:
            fn(self.store, rec, out)


# ============================================================================== figure
CLASS_COLORS = {"rule_error": "#8c8c8c", "data_defect": "#c0392b", "data_defect_handled": "#e0a030",
                "coverage_gap": "#2f7fc1", "not_reproduced": "#5aa469", "unreviewed": "#6c4f9e",
                "drilldown_error": "#000000"}
SEVERITY_ORDER = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3, "INFO": 4}


def fig_dashboard(store: StudyStore, rec: FigureRecord, out: dict[str, pl.DataFrame]) -> None:
    from matplotlib.patches import Patch

    from ..figkit import empty_panel, figure, finish

    d = (out[DQ_FAILURES.name]
         .with_columns(pl.col("severity").replace_strict(SEVERITY_ORDER, default=9, return_dtype=pl.Int64).alias("_o"))
         .sort("_o", "check_id").drop("_o"))
    fig, ax = figure(1, 2)
    a, b = ax[0, 0], ax[0, 1]
    if not d.height:
        empty_panel(a, "no failed checks")
        empty_panel(b, "no failed checks")
    else:
        y = np.arange(d.height)
        labels = [f"[{s}] {c}" for s, c in zip(d["severity"].to_list(), d["check_id"].to_list())]
        rows = d["n_affected_rows"].fill_null(0).to_numpy().astype(float)
        colors = [CLASS_COLORS.get(c, "#000000") for c in d["classification"].to_list()]
        a.barh(y, np.maximum(rows, 0.8), color=colors)
        a.set_xscale("log")
        a.set_yticks(y, labels, fontsize=7)
        a.invert_yaxis()
        a.set_xlabel("affected athlete rows (exact; log scale)")
        xmax = a.get_xlim()[1]
        for yi, (m, s) in enumerate(zip(d["drilldown_mode"].to_list(), d["phase01_sampled"].to_list())):
            a.text(xmax, yi, f" {m}{' · P01 sampled' if s else ''}", va="center", fontsize=7)
        a.legend(handles=[Patch(color=v, label=k) for k, v in CLASS_COLORS.items()
                          if k in set(d["classification"].to_list())], frameon=False, fontsize=7, loc="lower right")
        ath = d["n_affected_athletes"].fill_null(0).to_numpy().astype(float)
        lnk = d["n_affected_linked_athletes"].fill_null(0).to_numpy().astype(float)
        b.barh(y - 0.2, ath, height=0.4, color="#2f7fc1", label="athletes")
        b.barh(y + 0.2, lnk, height=0.4, color="#5aa469", label="Combine-linked athletes")
        b.set_yticks(y, [""] * d.height)
        b.invert_yaxis()
        b.set_xlabel("affected athletes")
        b.legend(frameon=False, fontsize=8)
    finish(fig, "Every non-PASS Phase 01 check re-measured exactly on bronze (trk checks: all M1 trajectories). "
                "Bar colour = diagnostic classification (a hypothesis for review, not a verdict). Zero-row bars are "
                "drawn at 0.8 on the log axis.", "Data-quality failures: what is affected, and why")
    store.save_figure(fig, rec, {"failures": d.select("check_id", "severity", "classification", "drilldown_mode",
                                                      "n_affected_rows", "n_affected_trajectories",
                                                      "n_affected_athletes", "n_affected_linked_athletes", "blocking")},
                      n=f"{d.height} checks")


FIGURES = [(FigureRecord(
    figure_id="A0-01", name="A0_01_dq_failure_dashboard", study="A0",
    research_question="Which Phase 01 checks failed, how many rows / trajectories / athletes do they affect, and "
                      "is each a rule error, a data defect, a coverage gap or already handled?",
    expected_insight="separate blocking defects from wrong rules before any athlete ranking",
    statistic="exact affected counts", dataset=DQ_FAILURES.name, grain="check", filters="result ∈ FAIL/WARN/ERROR",
    method="exact DuckDB drill-down per check family", publication=False), fig_dashboard)]
