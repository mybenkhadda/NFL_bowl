"""Step 10 — research feasibility grounded in measured coverage.

Numbers (athlete counts per hypothesis and position) are computed from gold/silver tables.
Qualitative fields (confounders, biases, validation, interpretation) are design assumptions and
are labelled as such. Drill-name patterns are regexes to be confirmed against
``gold.dim_combine_drill`` — only one drill name (OL_PULL_DRILL_FOLD_BLOCK_RIGHT) had been
observed when this module was written.

Feasibility rule of thumb (documented, adjustable): matched athletes ≥ 150 → High,
50–149 → Medium, < 50 → Low (per primary position group).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import polars as pl

from .config import Phase01Config, duckdb_connect
from .frames import query_column, query_df
from .reporting import CATEGORICAL, INK, MarkdownReport, add_caption, export_table, new_figure, save_figure
from .tables import LayerRefs
from .utils import get_logger

log = get_logger(__name__)


@dataclass(frozen=True)
class Hypothesis:
    hid: str
    title: str
    positions: tuple[str, ...]          # matched against position_group / nfl_position
    drill_regex: str                    # Combine tracking drills that carry the predictor
    traditional: tuple[str, ...]        # traditional baseline columns
    nfl_outcome: str                    # description
    outcome_sql: str = ""               # optional: SQL returning nfl_id with an outcome available
    confounders: str = ""
    biases: str = ""
    validation: str = ""
    interpretation: str = ""
    limitations: str = ""


DL = ("DL", "DE", "DT", "EDGE", "IDL", "NT")
HYPOTHESES = [
    Hypothesis(
        "H01", "Combine acceleration signature transfer", ("WR", "DB", "CB", "S", "RB"),
        r"(?i)(40|DASH|SPRINT|SPLIT)", ("forty", "ten_yd_split"),
        "Context-adjusted in-game acceleration (play max_a / time-to-speed from frames)",
        confounders="Route/assignment mix, play context (down, distance), scheme, fatigue, injuries, age.",
        biases="Only drafted / rostered athletes appear in NFL tracking (survivorship); opportunity varies by depth chart.",
        validation="Grouped-by-athlete CV; cohort holdout (train 2023-24 cohorts, test 2025); compare vs 40/10-yd baseline.",
        interpretation="Does the *shape* of acceleration (not the 40 time) carry over to how quickly a player reaches speed in games?",
        limitations="In-game acceleration is rarely maximal; needs event windows (e.g., snap → +2 s) in Phase 02.",
    ),
    Hypothesis(
        "H02", "Combine agility vs in-game change of direction", ("WR", "DB", "CB", "S", "LB"),
        r"(?i)(CONE|SHUTTLE|AGILITY|L_DRILL|COD|TURN)", ("three_cone", "short_shuttle"),
        "Braking + re-acceleration around direction changes in game frames",
        confounders="Route tree / coverage assignment; defender leverage; game state.",
        biases="Many athletes skip agility drills (see participation heatmap) → non-random missingness.",
        validation="Athlete-grouped CV within position; sensitivity to direction-change threshold definitions.",
        interpretation="Is Combine change-of-direction quality visible in route breaks / coverage transitions?",
        limitations="Requires robust heading-change detection on noisy tracking (smoothing study first).",
    ),
    Hypothesis(
        "H03", "Defensive line explosiveness", DL,
        r"(?i)(DL|GET_OFF|BAG|EXPLOS|HOOP|RIP|SWIM)", ("ten_yd_split", "broad_jump", "vertical"),
        "player_get_off / time_to_pressure from player_play; first-step speed after snap from frames",
        outcome_sql="SELECT nfl_id, count(*) AS n FROM {player_play} WHERE player_get_off IS NOT NULL GROUP BY 1 HAVING count(*) >= 20",
        confounders="Alignment (lined_up_position), stunts, double teams, pass vs run, down & distance.",
        biases="Get-off only defined on pass-rush snaps; rotational players have fewer snaps.",
        validation="Athlete-grouped CV; adjust for alignment; bootstrap CIs clustered by athlete.",
        interpretation="Do Combine burst signatures predict how fast DL get off the ball in games?",
        limitations="Snap timing precision (10 Hz) bounds get-off resolution (~0.1 s).",
    ),
    Hypothesis(
        "H04", "Athletic outlier identification", (),
        r".*", ("ngs_athleticism_score", "forty", "broad_jump", "vertical"),
        "Career outcomes (snaps, starts) — targets only, with exposure-length adjustment",
        outcome_sql="SELECT nfl_id, 1 AS n FROM {career}",
        confounders="Draft capital (round/pick) strongly drives opportunity; team needs; injuries.",
        biases="Career length differs by cohort (2025 cohort has ≤1 season) → right-censoring.",
        validation="Within-cohort comparisons; control for draft pick; temporal holdout.",
        interpretation="Which athletes move unusually well relative to their traditional scores, and does it matter?",
        limitations="Outcome = opportunity, not pure performance; requires censoring-aware analysis.",
    ),
    Hypothesis(
        "H05", "Position-specific movement profiles", (),
        r".*", ("forty",),
        "Season movement distributions (mart_athlete_nfl_profile)",
        confounders="Role within position (slot vs outside WR, nickel vs boundary CB).",
        biases="Position labels differ between Combine and NFL (see dim_position crosswalk).",
        validation="Stability across seasons; split-half reliability of profiles.",
        interpretation="Do transfer relationships differ by position (pooled vs position-aware models)?",
        limitations="Small positions (K/P/LS) are not analysable; group positions sensibly.",
    ),
]


def rating(n: int) -> str:
    return "High" if n >= 150 else ("Medium" if n >= 50 else "Low")


class ResearchFeasibilityAssessor:
    def __init__(self, cfg: Phase01Config, con=None):
        self.cfg = cfg
        self.con = con or duckdb_connect(cfg)
        self.refs = LayerRefs(cfg)

    def _athletes_with_drill(self, regex: str) -> str:
        return (f"SELECT DISTINCT nfl_id FROM {self.refs.sql('gold', 'fact_combine_attempt')} "
                f"WHERE regexp_matches(drill_name, '{regex}')")

    def assess(self, h: Hypothesis) -> tuple[dict, pl.DataFrame]:
        dim = self.refs.sql("gold", "dim_athlete")
        rb = self.refs.sql("gold", "mart_athlete_research_base")
        pos_filter = "TRUE" if not h.positions else \
            "(position_group IN ({0}) OR nfl_position IN ({0}))".format(", ".join(f"'{p}'" for p in h.positions))
        trad_any = " OR ".join(f"cmb_{c} IS NOT NULL" for c in h.traditional)
        drill_ids = self._athletes_with_drill(h.drill_regex) if self.refs.exists("gold", "fact_combine_attempt") \
            else "SELECT NULL::BIGINT AS nfl_id WHERE FALSE"
        outcome = "SELECT nfl_id FROM " + dim + " WHERE has_nfl_tracking"
        if h.outcome_sql:
            src = {"player_play": self.refs.sql("silver", "player_play") if self.refs.exists("silver", "player_play") else None,
                   "career": self.refs.sql("silver", "player_career_successes") if self.refs.exists("silver", "player_career_successes") else None}
            if all(v is not None for k, v in src.items() if "{" + k + "}" in h.outcome_sql):
                outcome = f"SELECT nfl_id FROM ({h.outcome_sql.format(**{k: v for k, v in src.items() if v})})"
        q = f"""
            WITH pop AS (SELECT * FROM {rb} WHERE {pos_filter}),
                 drill AS ({drill_ids}), outc AS ({outcome})
            SELECT position_group, count(*) AS n_population,
                   count_if({trad_any}) AS n_traditional,
                   count_if(nfl_id IN (SELECT nfl_id FROM drill)) AS n_combine_tracking_drill,
                   count_if(nfl_id IN (SELECT nfl_id FROM outc)) AS n_outcome,
                   count_if(nfl_id IN (SELECT nfl_id FROM drill) AND nfl_id IN (SELECT nfl_id FROM outc)) AS n_matched,
                   count_if(nfl_id IN (SELECT nfl_id FROM drill) AND nfl_id IN (SELECT nfl_id FROM outc) AND ({trad_any}))
                       AS n_matched_with_baseline
            FROM pop GROUP BY 1 ORDER BY n_matched DESC
        """
        by_pos = query_df(self.con, q).with_columns(pl.lit(h.hid).alias("hypothesis"))
        tot = by_pos.select(pl.exclude("position_group", "hypothesis").sum()).row(0, named=True) if by_pos.height else {}
        drills = []
        if self.refs.exists("gold", "dim_combine_drill"):
            drills = query_column(self.con, f"SELECT drill_name FROM {self.refs.sql('gold', 'dim_combine_drill')} "
                                            f"WHERE regexp_matches(drill_name, '{h.drill_regex}') ORDER BY n_athletes DESC LIMIT 8")
        best_pos = by_pos.row(0, named=True) if by_pos.height else {"position_group": None, "n_matched": 0}
        row = {
            "hypothesis": h.hid, "title": h.title, "positions": ", ".join(h.positions) or "all",
            "drill_regex": h.drill_regex, "matching_drills": ", ".join(drills) or "none matched",
            "traditional_baseline": ", ".join(h.traditional), "nfl_outcome": h.nfl_outcome,
            **{k: int(v or 0) for k, v in tot.items()},
            "largest_position": best_pos["position_group"], "largest_position_n_matched": int(best_pos["n_matched"] or 0),
            "feasibility": rating(int(best_pos["n_matched"] or 0)),
            "confounders": h.confounders, "selection_bias": h.biases, "validation": h.validation,
            "football_interpretation": h.interpretation, "limitations": h.limitations,
        }
        return row, by_pos

    def run(self) -> tuple[pl.DataFrame, pl.DataFrame]:
        rows, details = [], []
        for h in HYPOTHESES:
            try:
                r, d = self.assess(h)
                rows.append(r)
                details.append(d)
            except Exception as exc:  # noqa: BLE001
                log.error("feasibility %s failed: %s", h.hid, exc)
                rows.append({"hypothesis": h.hid, "title": h.title, "feasibility": "BLOCKED", "limitations": str(exc)})
        matrix = pl.DataFrame(rows, infer_schema_length=None)
        detail = pl.concat(details, how="diagonal_relaxed") if details else pl.DataFrame()
        export_table(matrix, self.cfg.paths.tables, "research_feasibility_matrix")
        if not detail.is_empty():
            export_table(detail, self.cfg.paths.tables, "research_feasibility_by_position")
        return matrix, detail

    def plot(self, detail: pl.DataFrame) -> Path | None:
        if detail.is_empty():
            return None
        hs = detail["hypothesis"].unique(maintain_order=True).to_list()
        fig, axes = new_figure(1, len(hs), sharey=False, squeeze=False)
        for ax, h in zip(axes[0], hs):
            d = detail.filter((pl.col("hypothesis") == h) & (pl.col("n_population") > 0)).sort("n_matched").tail(8)
            ax.barh(d["position_group"].to_list(), d["n_population"].to_list(), color="#cde2fb", height=0.7, label="population")
            ax.barh(d["position_group"].to_list(), d["n_matched"].to_list(), color=CATEGORICAL[0], height=0.4,
                    label="matched (drill ∧ outcome)")
            ax.axvline(50, color=INK["muted"], linestyle=":", linewidth=1)
            ax.axvline(150, color=INK["muted"], linestyle="--", linewidth=1)
            ax.set_title(h, fontsize=12)
            ax.grid(axis="y", visible=False)
        axes[0, 0].legend(loc="lower right", fontsize=8)
        add_caption(fig, "Athletes per position. Dotted = 50 (Medium), dashed = 150 (High) matched-athlete thresholds.")
        return save_figure(fig, self.cfg.paths.figures, "08_feasibility_by_position")

    def write_report(self, matrix: pl.DataFrame, detail: pl.DataFrame) -> Path:
        p = self.cfg.paths
        fig = self.plot(detail)
        rpt = MarkdownReport("08 — Research feasibility", "Step 10: hypotheses vs. measured data coverage")
        rpt.callout("NOTE", "Counts are measured from gold tables. Confounders, biases, validation and interpretation "
                    "are design assumptions to challenge, not findings. No model results are reported here.")
        if fig:
            rpt.image(fig, p.reports_root, "feasibility")
        cols = [c for c in ("hypothesis", "title", "positions", "n_population", "n_traditional", "n_combine_tracking_drill",
                            "n_outcome", "n_matched", "n_matched_with_baseline", "largest_position",
                            "largest_position_n_matched", "feasibility") if c in matrix.columns]
        rpt.h2("Feasibility matrix").table(matrix.select(cols))
        for r in matrix.iter_rows(named=True):
            rpt.h3(f"{r['hypothesis']} — {r.get('title')}")
            rpt.bullets([f"**{k.replace('_', ' ').capitalize()}:** {r.get(k)}" for k in (
                "drill_regex", "matching_drills", "traditional_baseline", "nfl_outcome", "confounders", "selection_bias",
                "validation", "football_interpretation", "limitations") if r.get(k) is not None])
        rpt.h2("Per-position detail").table(detail, max_rows=120) if not detail.is_empty() else None
        return rpt.save(p.reports_root / "08_research_feasibility.md")
