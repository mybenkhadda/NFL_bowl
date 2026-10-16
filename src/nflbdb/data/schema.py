"""Step 2 — schema discovery and semantic dictionary.

Evidence levels used in the catalog:

* ``confirmed``  — directly observed (column exists, physical type parses, values seen).
* ``inferred``   — interpretation consistent with observed values and standard NFL / NGS
                   conventions, NOT yet confirmed against the official Kaggle data page.
* ``unresolved`` — meaning or unit unclear; listed as an open question in the report.

The semantic dictionary is keyed by column name. A shared column name across tables is a
*hypothesis* of shared meaning; namespace equality is tested in :mod:`relationships`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import polars as pl

from .config import Phase01Config
from .csv_schema import SchemaResolver
from .reporting import MarkdownReport, export_table
from .sampling import head_sample, offset_sample
from .sources import SOURCES, SourceSpec
from .utils import get_logger

log = get_logger(__name__)


@dataclass(frozen=True)
class Semantic:
    role: str            # identifier | foreign_key | timestamp | categorical | continuous | kinematic | outcome | text | flag | count
    meaning: str
    unit: str = ""
    evidence: str = "inferred"


S = Semantic
SEMANTICS: dict[str, Semantic] = {
    # identifiers -----------------------------------------------------------------
    "nfl_id": S("identifier", "Player identifier; appears in every player-bearing table", evidence="confirmed"),
    "game_id": S("identifier", "Game identifier; format YYYYMMDDNN (date + sequence) observed", evidence="confirmed"),
    "play_id": S("identifier", "Play identifier, unique within a game (hypothesis)"),
    "event_id": S("identifier", "Combine tracking session/event id; observed format "
                  "'<yyyymmddhhmmss>_<epoch_ms>_<seq>'", evidence="confirmed"),
    "game_key": S("identifier", "Alternate game key (league internal)", evidence="unresolved"),
    # time ------------------------------------------------------------------------
    "time": S("timestamp", "Frame timestamp, ISO-8601 with milliseconds; 100 ms steps in sample", "UTC? (unresolved)"),
    "draft_year": S("categorical", "NFL draft cohort year (= Combine year)", "year"),
    "season": S("categorical", "NFL season (season 2023 includes Jan-2024 games)", "year", "confirmed"),
    "week": S("categorical", "Season week", "week"),
    "season_type": S("categorical", "REG / POST", evidence="confirmed"),
    "birth_date": S("timestamp", "Date of birth", "date"),
    "game_clock": S("text", "Game clock at snap, 'MM:SS'", "mm:ss", "confirmed"),
    "quarter": S("categorical", "Quarter (5 = OT, hypothesis)"),
    # tracking kinematics ----------------------------------------------------------
    "x": S("kinematic", "Position along the long axis of the field", "yards"),
    "y": S("kinematic", "Position across the field", "yards"),
    "s": S("kinematic", "Speed (vendor-computed, possibly smoothed)", "yards/second"),
    "a": S("kinematic", "Acceleration magnitude (vendor-computed)", "yards/second²"),
    "dis": S("kinematic", "Distance travelled since previous frame", "yards"),
    "o": S("kinematic", "Body orientation", "degrees"),
    "dir": S("kinematic", "Direction of motion", "degrees"),
    "event": S("categorical", "Tracking event tag (snap, pass_forward, …); sparse"),
    "entity_type": S("categorical", "Tracked entity: PLAYER or BALL. BALL rows carry the athlete's nfl_id "
                     "(ball in that athlete's rep), so entity_type is part of the frame grain", evidence="confirmed"),
    "drill_type": S("categorical", "Combine drill family (e.g. SKILL_DRILLS_OL)", evidence="confirmed"),
    "drill_name": S("categorical", "Specific drill (e.g. OL_PULL_DRILL_FOLD_BLOCK_RIGHT)", evidence="confirmed"),
    "attempt": S("count", "Attempt / repetition number within a drill", "count"),
    # combine results ---------------------------------------------------------------
    "combine_position": S("categorical", "Position group at the Combine (e.g. DB, WR, TE)", evidence="confirmed"),
    "combine_height": S("continuous", "Height (77.875 observed → fractional inches)", "inches"),
    "combine_weight": S("continuous", "Weight", "pounds"),
    "hand_size": S("continuous", "Hand span", "inches"),
    "arm_length": S("continuous", "Arm length", "inches"),
    "wing_span": S("continuous", "Wingspan", "inches"),
    "ten_yd_split": S("continuous", "10-yard split of the 40-yard dash", "seconds"),
    "forty": S("continuous", "40-yard dash time", "seconds"),
    "vertical": S("continuous", "Vertical jump", "inches"),
    "broad_jump": S("continuous", "Broad jump", "inches"),
    "three_cone": S("continuous", "3-cone drill time", "seconds"),
    "short_shuttle": S("continuous", "20-yard (short) shuttle time", "seconds"),
    "bench_reps": S("count", "225-lb bench press repetitions", "reps"),
    "ngs_athleticism_score": S("continuous", "NGS athleticism score (pre-draft)", "score 0-100"),
    "ngs_college_production_score": S("continuous", "NGS college production score", "score 0-100"),
    "ngs_final_score": S("continuous", "NGS overall draft score", "score 0-100"),
    # players -----------------------------------------------------------------------
    "display_name": S("text", "Player name — never used as a join key", evidence="confirmed"),
    "nfl_position": S("categorical", "Listed NFL position", evidence="confirmed"),
    "college_name": S("categorical", "College"),
    "college_conference": S("categorical", "College conference"),
    "draft_round": S("continuous", "Draft round (null = undrafted? unresolved)", "round"),
    "draft_pick_within_round": S("continuous", "Pick within round", "pick"),
    "draft_overall_pick": S("continuous", "Overall pick", "pick"),
    # outcomes ----------------------------------------------------------------------
    **{c: S("outcome", f"Career outcome: {c.replace('_', ' ')}", "count") for c in SOURCES["player_career_successes"].columns if c != "nfl_id"},
    # player_play — play context ------------------------------------------------------
    "team_abbr": S("categorical", "Player's team on the play", evidence="confirmed"),
    "possession_team": S("categorical", "Offensive team", evidence="confirmed"),
    "lined_up_position": S("categorical", "Alignment on the play (T, EDGE, OLB, INTERIOR_LINE observed)", evidence="confirmed"),
    "play_description": S("text", "Play-by-play text", evidence="confirmed"),
    "down": S("categorical", "Down"), "yards_to_go": S("continuous", "Yards to first down", "yards"),
    "yardline_side": S("categorical", "Side of field of the line of scrimmage"),
    "yardline_number": S("continuous", "Yard line number", "yards"),
    "play_direction": S("categorical", "Offense direction of travel (left/right) — drives coordinate standardisation", evidence="confirmed"),
    "offense_formation": S("categorical", "Offensive formation"), "receiver_alignment": S("categorical", "Receiver alignment (e.g. 3x1)"),
    "pass_result": S("categorical", "Pass result code (I = incomplete, R = run? unresolved)", evidence="unresolved"),
    "team_coverage_man_zone": S("categorical", "Defensive coverage family"), "team_coverage_type": S("categorical", "Coverage shell"),
    "route_ran": S("categorical", "Route run by the player"),
    "coverage_assignment": S("categorical", "Defender coverage assignment"),
    "coverage_assignment_at_snap": S("categorical", "Defender assignment at snap"),
    "play_nullified_by_penalty": S("flag", "Y/N penalty nullification", evidence="confirmed"),
}
_FLAG_COLS = ["in_motion_at_ball_snap", "target", "pressure_allowed", "quick_pressure", "unblocked_pressure",
              "blitzing", "tackle_for_loss"]
for _c in _FLAG_COLS:
    SEMANTICS.setdefault(_c, S("flag", f"Player-level flag: {_c.replace('_', ' ')} (TRUE/FALSE literals)"))
for _c in ["sack", "tackle", "assist", "caused_forced_fumble", "recovered_fumble", "fumble", "fumble_lost",
           "sack_allowed", "extended_sack_allowed", "pass_rushers_encountered"]:
    SEMANTICS.setdefault(_c, S("outcome", f"Player-play event count: {_c.replace('_', ' ')}", "count"))
for _c in ["time_to_pressure", "time_to_qb_hurry", "time_to_pressure_allowed", "dropback_duration", "player_get_off"]:
    SEMANTICS.setdefault(_c, S("continuous", f"Timing metric: {_c.replace('_', ' ')}", "seconds"))
for _c in ["rec_yards", "rush_yards", "yards_after_catch", "expected_yards_after_catch", "yards_gained",
           "pre_penalty_yards_gained", "penalty_yards", "pass_length", "separation_at_pass_forward", "cushion"]:
    SEMANTICS.setdefault(_c, S("continuous", f"Distance metric: {_c.replace('_', ' ')}", "yards"))


def semantic_for(column: str) -> Semantic:
    if column in SEMANTICS:
        return SEMANTICS[column]
    if "probability" in column or "probility" in column:
        return S("continuous", f"Model probability: {column.replace('_', ' ')}", "probability 0-1")
    if "expected_points" in column:
        return S("continuous", "Expected points (added)", "points")
    if column.endswith("_score"):
        return S("continuous", "Score", "points")
    return S("unknown", "No interpretation yet", evidence="unresolved")


GRAIN_HYPOTHESES = {
    "players": "One row per player (nfl_id).",
    "combine_results": "One row per athlete per Combine (nfl_id, possibly + draft_year).",
    "combine_tracking": "One row per tracked entity (PLAYER or BALL) per 100 ms frame within a Combine drill attempt "
                        "(event_id, nfl_id, attempt, entity_type, time). Hierarchy: draft_year → athlete → drill → "
                        "attempt → entity → frame.",
    "game_tracking": "One row per tracked entity per 100 ms frame within a play (game_id, play_id, nfl_id, time). "
                     "Rows with null nfl_id are expected to be the football. Hierarchy: season → game → play → player → frame.",
    "games": "One row per game.",
    "player_play": "One row per player per play; play-level columns repeated on every player row.",
    "player_career_successes": "One row per player: career totals up to the data cut-off (outcome table).",
}


class SchemaCatalog:
    def __init__(self, cfg: Phase01Config):
        self.cfg = cfg

    def _sample(self, spec: SourceSpec) -> pl.DataFrame:
        files = spec.files(self.cfg.paths.data_root)
        if not files:
            return pl.DataFrame()
        pc = self.cfg.profiling
        frames = []
        for f in files:
            df = offset_sample(f, spec, pc.offset_sample_chunks, pc.offset_sample_lines, pc.seed)
            if df.is_empty():
                df = head_sample(f, spec, pc.schema_sample_rows)
            frames.append(df)
        return pl.concat(frames, how="vertical_relaxed")

    def build(self) -> tuple[pl.DataFrame, dict[str, pl.DataFrame]]:
        """Returns (catalog, samples). Samples are kept so profiling can reuse them in-memory."""
        rows, samples = [], {}
        col_tables: dict[str, list[str]] = {}
        for spec in SOURCES.values():
            for c in spec.columns:
                col_tables.setdefault(c, []).append(spec.name)
        for spec in SOURCES.values():
            sample = self._sample(spec)
            samples[spec.name] = sample
            if sample.is_empty():
                log.warning("no sample for %s (file missing?)", spec.name)
                continue
            key_parts = {c for k in spec.grain_candidates for c in k}
            # Sample holds raw strings: test each declared type against the observed values.
            diag = {r.column: r for r in SchemaResolver().diagnose(sample, spec, f"bounded sample ({sample.height:,} rows)")}
            for i, col in enumerate(spec.columns):
                sem = semantic_for(col)
                s = sample[col] if col in sample.columns else pl.Series(col, [], dtype=pl.String)
                non_null = s.drop_nulls()
                examples = non_null.unique(maintain_order=True).head(5).cast(pl.String).to_list()
                d = diag.get(col)
                rows.append({
                    "table": spec.name, "column": col, "ordinal": i,
                    "declared_dtype": str(spec.schema[col]),
                    "bronze_dtype": d.resolved if d else str(spec.schema[col]),
                    "type_resolution": d.action if d else "string",
                    "sample_cast_failures": d.n_cast_fail if d else 0,
                    "sample_fail_examples": d.fail_examples if d else [],
                    "sample_empty_strings": d.n_empty if d else int((non_null.str.strip_chars() == "").sum()),
                    "semantic_role": sem.role, "meaning": sem.meaning, "unit": sem.unit, "evidence": sem.evidence,
                    "sample_rows": sample.height,
                    "sample_null_frac": round(s.null_count() / max(len(s), 1), 4),
                    "sample_n_distinct": non_null.n_unique(),
                    "sample_examples": examples,
                    "is_candidate_key_part": col in key_parts,
                    "shared_with": [t for t in col_tables[col] if t != spec.name],
                    "grain_hypothesis": GRAIN_HYPOTHESES.get(spec.name, ""),
                })
        return pl.DataFrame(rows, infer_schema_length=None), samples

    def export(self, catalog: pl.DataFrame) -> list[Path]:
        return export_table(catalog, self.cfg.paths.tables, "schema_catalog")

    def write_report(self, catalog: pl.DataFrame) -> Path:
        p = self.cfg.paths
        rpt = MarkdownReport("02 — Schema dictionary", "Step 2: physical schema, semantics and grain hypotheses")
        rpt.p("Samples are deterministic byte-offset samples spanning each file (see `sampling.offset_sample`). "
              "Null fractions and distinct counts below are **sample statistics**, not full-table values.")
        rpt.h2("Grain hypotheses (validated in 03_data_quality.md)")
        rpt.table(pl.DataFrame({"table": list(GRAIN_HYPOTHESES), "grain": list(GRAIN_HYPOTHESES.values())}))
        rpt.h2("Type resolution (raw values read as strings, declared types tested)")
        rpt.p("`declared_dtype` comes from `sources.py`; `bronze_dtype` is what the sample supports. Columns whose "
              "values are only TRUE/FALSE stay String in bronze and become BOOLEAN in silver. The bronze build "
              "re-tests non-tracking tables on a full scan (`reports/tables/schema_resolution.csv`).")
        changed = catalog.filter(pl.col("type_resolution").is_in(["boolean_like", "mixed_numeric_boolean", "error"])
                                 | (pl.col("sample_empty_strings") > 0))
        rpt.table(changed.select("table", "column", "declared_dtype", "bronze_dtype", "type_resolution",
                                 "sample_cast_failures", "sample_fail_examples", "sample_empty_strings")) \
            if changed.height else rpt.p("_All declared types are consistent with the sample._")
        rpt.h2("Identifiers shared across tables (namespace equality NOT assumed)")
        shared = (catalog.filter(pl.col("semantic_role") == "identifier")
                  .group_by("column").agg(pl.col("table").sort().alias("tables")).sort("column"))
        rpt.table(shared)
        for table in catalog["table"].unique(maintain_order=True):
            t = catalog.filter(pl.col("table") == table)
            rpt.h2(f"`{table}` — {t.height} columns")
            rpt.table(t.select("column", "declared_dtype", "bronze_dtype", "semantic_role", "unit", "evidence", "sample_null_frac",
                               "sample_n_distinct", "sample_examples", "meaning"), max_rows=100)
        unresolved = catalog.filter(pl.col("evidence") == "unresolved")
        rpt.unresolved([f"`{r['table']}.{r['column']}` — {r['meaning']}" for r in unresolved.iter_rows(named=True)] + [
            "Timezone of `time` columns (UTC vs stadium local) — compare against kickoff times.",
            "Does `combine_tracking.event_id` identify one attempt or a whole session? (tested via key uniqueness).",
            "Coordinate frame of Combine tracking (field-relative like game tracking?).",
            "Are `s`/`a` smoothed by the vendor? (tested in tracking QC: reported vs. derived speed).",
        ])
        return rpt.save(p.reports_root / "02_schema_dictionary.md")
