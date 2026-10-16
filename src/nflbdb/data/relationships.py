"""Step 4 — relationships, athlete linkage and join cardinality.

Linkage policy: athletes are linked across sources ONLY by exact ``nfl_id`` equality, and that
equality is itself tested (value ranges, overlap, conflicting attributes). Names are used only
as a diagnostic for potential duplicate identities — never as a join key.

Join-size estimates use per-key counts on each side:  Σ_k count_L(k) · count_R(k). No large
join is materialised to obtain them.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import polars as pl

from .config import Phase01Config, duckdb_connect
from .frames import query_df, query_row, query_scalar
from .reporting import (CATEGORICAL, INK, SEQUENTIAL_BLUE, MarkdownReport, add_caption, export_table, new_figure,
                        save_figure)
from .tables import LayerRefs
from .utils import atomic_write_text, get_logger

log = get_logger(__name__)

# nfl_id sources and the predicate that restricts them to athletes.
ID_SOURCES: dict[str, str] = {
    "players": "TRUE",
    "combine_results": "TRUE",
    "combine_tracking": "entity_type = 'PLAYER'",
    "game_tracking": "TRUE",
    "player_play": "TRUE",
    "player_career_successes": "TRUE",
}
SHORT = {"players": "players", "combine_results": "cmb_results", "combine_tracking": "cmb_tracking",
         "game_tracking": "nfl_tracking", "player_play": "player_play", "player_career_successes": "career"}


@dataclass(frozen=True)
class JoinPlan:
    left: str
    right: str
    keys: tuple[str, ...]
    purpose: str
    left_filter: str = "TRUE"
    right_filter: str = "TRUE"


JOIN_PLANS = [
    JoinPlan("combine_results", "players", ("nfl_id",), "Attach biography to traditional Combine results"),
    JoinPlan("combine_tracking", "combine_results", ("nfl_id",), "Tracking frames ↔ traditional results",
             "entity_type = 'PLAYER'"),
    JoinPlan("player_play", "games", ("game_id",), "Play rows ↔ game metadata"),
    JoinPlan("game_tracking", "player_play", ("game_id", "play_id", "nfl_id"), "Frames ↔ player-play context"),
    JoinPlan("game_tracking", "player_play", ("game_id", "play_id"),
             "ANTI-PATTERN: frames ↔ player_play on play only (every frame × every player)"),
    JoinPlan("game_tracking", "players", ("nfl_id",), "Frames ↔ biography"),
    JoinPlan("combine_tracking", "game_tracking", ("nfl_id",),
             "ANTI-PATTERN: Combine frames × NFL frames — aggregate both sides to athlete level first",
             "entity_type = 'PLAYER'"),
]


class RelationshipAnalyzer:
    def __init__(self, cfg: Phase01Config, con=None, layer: str = "bronze"):
        self.cfg = cfg
        self.con = con or duckdb_connect(cfg)
        self.refs = LayerRefs(cfg)
        self.layer = layer
        self._ids_built = False

    def ref(self, t: str) -> str:
        return self.refs.sql(self.layer, t)

    # ------------------------------------------------------------------ id sets
    def build_id_sets(self) -> None:
        for src, pred in ID_SOURCES.items():
            if not self.refs.exists(self.layer, src):
                self.con.execute(f"CREATE OR REPLACE TEMP TABLE ids_{src} (nfl_id BIGINT)")
                log.warning("id set empty: %s not built", src)
                continue
            self.con.execute(
                f"CREATE OR REPLACE TEMP TABLE ids_{src} AS "
                f"SELECT DISTINCT nfl_id FROM {self.ref(src)} WHERE nfl_id IS NOT NULL AND {pred}"
            )
        self._ids_built = True

    def id_namespace(self) -> pl.DataFrame:
        """Value ranges per source — disjoint ranges would reveal different id namespaces."""
        self._ensure_ids()
        parts = [f"SELECT '{s}' AS source, count(*) AS n_ids, min(nfl_id) AS min_id, max(nfl_id) AS max_id, "
                 f"median(nfl_id) AS median_id FROM ids_{s}" for s in ID_SOURCES]
        return query_df(self.con, " UNION ALL ".join(parts))

    def overlap_matrix(self) -> pl.DataFrame:
        self._ensure_ids()
        rows = []
        for a in ID_SOURCES:
            for b in ID_SOURCES:
                n = query_scalar(self.con, f"SELECT count(*) FROM ids_{a} JOIN ids_{b} USING (nfl_id)")
                rows.append({"source_a": a, "source_b": b, "n_common": n})
        return pl.DataFrame(rows)

    def _ensure_ids(self) -> None:
        if not self._ids_built:
            self.build_id_sets()

    # ------------------------------------------------------------------ athlete linkage
    def athlete_presence(self) -> pl.DataFrame:
        """One row per nfl_id seen anywhere, with source flags, attributes and NFL exposure."""
        self._ensure_ids()
        union = " UNION ".join(f"SELECT nfl_id FROM ids_{s}" for s in ID_SOURCES)
        flags = ",\n".join(f"a.nfl_id IN (SELECT nfl_id FROM ids_{s}) AS in_{SHORT[s]}" for s in ID_SOURCES)
        have = lambda t: self.refs.exists(self.layer, t)  # noqa: E731
        players = (f"(SELECT nfl_id, any_value(display_name) AS display_name, any_value(nfl_position) AS nfl_position, "
                   f"any_value(draft_year) AS players_draft_year, count(*) AS n_player_rows FROM {self.ref('players')} GROUP BY 1)"
                   if have("players") else "(SELECT NULL::BIGINT nfl_id, NULL display_name, NULL nfl_position, "
                                           "NULL::BIGINT players_draft_year, 0 n_player_rows WHERE FALSE)")
        combine = (f"(SELECT nfl_id, any_value(combine_position) AS combine_position, min(draft_year) AS combine_draft_year, "
                   f"count(*) AS n_combine_rows FROM {self.ref('combine_results')} GROUP BY 1)"
                   if have("combine_results") else "(SELECT NULL::BIGINT nfl_id, NULL combine_position, "
                                                   "NULL::BIGINT combine_draft_year, 0 n_combine_rows WHERE FALSE)")
        exposure = (f"(SELECT nfl_id, count(DISTINCT season) AS nfl_tracked_seasons, count(DISTINCT game_id) AS nfl_tracked_games, "
                    f"count(DISTINCT game_id * 100000 + play_id) AS nfl_tracked_plays FROM {self.ref('game_tracking')} "
                    f"WHERE nfl_id IS NOT NULL GROUP BY 1)"
                    if have("game_tracking") else "(SELECT NULL::BIGINT nfl_id, 0 nfl_tracked_seasons, 0 nfl_tracked_games, "
                                                  "0 nfl_tracked_plays WHERE FALSE)")
        q = f"""
            WITH a AS ({union})
            SELECT a.nfl_id, {flags},
                   p.display_name, p.nfl_position, p.players_draft_year, coalesce(p.n_player_rows, 0) AS n_player_rows,
                   c.combine_position, c.combine_draft_year, coalesce(c.n_combine_rows, 0) AS n_combine_rows,
                   coalesce(e.nfl_tracked_seasons, 0) AS nfl_tracked_seasons,
                   coalesce(e.nfl_tracked_games, 0) AS nfl_tracked_games,
                   coalesce(e.nfl_tracked_plays, 0) AS nfl_tracked_plays,
                   coalesce(c.combine_position, p.nfl_position, 'UNKNOWN') AS position_group,
                   coalesce(c.combine_draft_year, p.players_draft_year) AS cohort
            FROM a
            LEFT JOIN {players} p USING (nfl_id)
            LEFT JOIN {combine} c USING (nfl_id)
            LEFT JOIN {exposure} e USING (nfl_id)
        """
        return query_df(self.con, q)

    @staticmethod
    def linkage_funnel(presence: pl.DataFrame) -> pl.DataFrame:
        steps = [
            ("Combine results", pl.col("in_cmb_results")),
            ("+ Combine tracking", pl.col("in_cmb_results") & pl.col("in_cmb_tracking")),
            ("+ NFL game tracking", pl.col("in_cmb_results") & pl.col("in_cmb_tracking") & pl.col("in_nfl_tracking")),
            ("Combine tracking (any)", pl.col("in_cmb_tracking")),
            ("Combine tracking ∧ NFL tracking", pl.col("in_cmb_tracking") & pl.col("in_nfl_tracking")),
            ("NFL tracking (any)", pl.col("in_nfl_tracking")),
            ("NFL tracking ∧ career outcomes", pl.col("in_nfl_tracking") & pl.col("in_career")),
        ]
        return pl.DataFrame({"stage": [s for s, _ in steps],
                             "n_athletes": [presence.filter(e).height for _, e in steps]})

    @staticmethod
    def coverage_by(presence: pl.DataFrame, by: str) -> pl.DataFrame:
        return (
            presence.filter(pl.col("in_cmb_results") | pl.col("in_cmb_tracking"))
            .group_by(by)
            .agg(
                pl.len().alias("n_combine_athletes"),
                pl.col("in_cmb_tracking").sum().alias("n_with_combine_tracking"),
                pl.col("in_nfl_tracking").sum().alias("n_with_nfl_tracking"),
                (pl.col("in_cmb_tracking") & pl.col("in_nfl_tracking")).sum().alias("n_matched_tracking_both"),
                pl.col("in_career").sum().alias("n_with_career_outcomes"),
                pl.col("nfl_tracked_plays").filter(pl.col("in_nfl_tracking")).median().alias("median_nfl_plays_if_tracked"),
            )
            .with_columns(
                (pl.col("n_matched_tracking_both") / pl.col("n_combine_athletes")).round(3).alias("match_rate"),
            )
            .sort(by, nulls_last=True)
        )

    def ambiguity_checks(self) -> pl.DataFrame:
        """Diagnostics only: same name + birth date under different nfl_id; nfl_id with several
        combine rows or several names."""
        out = []
        if self.refs.exists(self.layer, "players"):
            out.append(query_df(self.con, f"""
                SELECT 'same name+birth_date, different nfl_id' AS issue, display_name || ' / ' || birth_date AS subject,
                       list(nfl_id) AS nfl_ids
                FROM {self.ref('players')} GROUP BY display_name, birth_date HAVING count(DISTINCT nfl_id) > 1
            """))
            out.append(query_df(self.con, f"""
                SELECT 'nfl_id with several names' AS issue, CAST(nfl_id AS VARCHAR) AS subject, list(DISTINCT display_name) AS nfl_ids
                FROM {self.ref('players')} GROUP BY nfl_id HAVING count(DISTINCT display_name) > 1
            """).with_columns(pl.col("nfl_ids").cast(pl.List(pl.Utf8))))
        if self.refs.exists(self.layer, "combine_results"):
            out.append(query_df(self.con, f"""
                SELECT 'nfl_id with several combine rows' AS issue, CAST(nfl_id AS VARCHAR) AS subject,
                       list(draft_year) AS nfl_ids
                FROM {self.ref('combine_results')} GROUP BY nfl_id HAVING count(*) > 1
            """))
        out = [o.with_columns(pl.col("nfl_ids").cast(pl.List(pl.Utf8))) for o in out if not o.is_empty()]
        return pl.concat(out, how="vertical_relaxed") if out else pl.DataFrame(
            schema={"issue": pl.Utf8, "subject": pl.Utf8, "nfl_ids": pl.List(pl.Utf8)})

    # ------------------------------------------------------------------ cardinality
    def join_cardinality(self, plans: list[JoinPlan] | None = None) -> pl.DataFrame:
        rows = []
        for jp in plans or JOIN_PLANS:
            if not (self.refs.exists(self.layer, jp.left) and self.refs.exists(self.layer, jp.right)):
                continue
            k = ", ".join(jp.keys)
            notnull = " AND ".join(f"{c} IS NOT NULL" for c in jp.keys)
            on = " AND ".join(f"l.{c} = r.{c}" for c in jp.keys)
            q = f"""
                WITH l AS (SELECT {k}, count(*) AS n FROM {self.ref(jp.left)} WHERE {notnull} AND {jp.left_filter} GROUP BY ALL),
                     r AS (SELECT {k}, count(*) AS n FROM {self.ref(jp.right)} WHERE {notnull} AND {jp.right_filter} GROUP BY ALL)
                SELECT (SELECT CAST(sum(n) AS BIGINT) FROM l) AS n_left, (SELECT CAST(sum(n) AS BIGINT) FROM r) AS n_right,
                       (SELECT count(*) FROM l) AS keys_left, (SELECT count(*) FROM r) AS keys_right,
                       (SELECT max(n) FROM l) AS max_per_key_left, (SELECT max(n) FROM r) AS max_per_key_right,
                       (SELECT count(*) FROM l JOIN r ON {on}) AS matched_keys,
                       (SELECT CAST(sum(CAST(l.n AS DOUBLE) * r.n) AS DOUBLE) FROM l JOIN r ON {on}) AS est_inner_rows,
                       (SELECT CAST(sum(n) AS BIGINT) FROM l ANTI JOIN r ON {on}) AS left_rows_unmatched
            """
            r = query_row(self.con, q)
            ml, mr = r["max_per_key_left"] or 0, r["max_per_key_right"] or 0
            kind = {(True, True): "1:1", (False, True): "N:1", (True, False): "1:N", (False, False): "N:M"}[(ml <= 1, mr <= 1)]
            est = float(r["est_inner_rows"] or 0)
            factor = est / max(r["n_left"] or 1, r["n_right"] or 1)
            rows.append({
                "left": jp.left, "right": jp.right, "keys": k, "purpose": jp.purpose, **r,
                "cardinality": kind, "multiplicative_factor": round(factor, 3),
                "left_match_rate": round(1 - (r["left_rows_unmatched"] or 0) / max(r["n_left"] or 1, 1), 4),
                "explosion_flag": factor > 1.5,
            })
            log.info("cardinality %s ⨝ %s on (%s): %s, est %.3g rows (×%.2f)", jp.left, jp.right, k, kind, est, factor)
        return pl.DataFrame(rows)

    # ------------------------------------------------------------------ figures
    def plot_funnel(self, funnel: pl.DataFrame) -> Path:
        fig, ax = new_figure(figsize=(12.8, 5.4))
        d = funnel.reverse()
        bars = ax.barh(d["stage"].to_list(), d["n_athletes"].to_list(), color=CATEGORICAL[0], height=0.6)
        for b, n in zip(bars, d["n_athletes"]):
            ax.text(b.get_width(), b.get_y() + b.get_height() / 2, f"  {n:,}", va="center", fontsize=10,
                    color=INK["secondary"])
        ax.set_xlabel("Athletes (distinct nfl_id, exact match only)")
        ax.set_title("Athlete linkage: who can be studied across Combine and NFL tracking")
        ax.grid(axis="y", visible=False)
        add_caption(fig, f"Source: {self.layer} layer, exact nfl_id equality. No name-based matching.")
        return save_figure(fig, self.cfg.paths.figures, "04_athlete_linkage_funnel")

    def plot_coverage_by_position(self, cov: pl.DataFrame) -> Path | None:
        if cov.is_empty():
            return None
        d = cov.sort("n_combine_athletes")
        fig, ax = new_figure()
        y = list(range(d.height))
        ax.barh(y, d["n_combine_athletes"].to_list(), color="#cde2fb", height=0.7, label="Combine athletes")
        ax.barh(y, d["n_with_nfl_tracking"].to_list(), color=CATEGORICAL[0], height=0.42, label="… with NFL tracking")
        ax.barh(y, d["n_matched_tracking_both"].to_list(), color=CATEGORICAL[1], height=0.2,
                label="… with Combine AND NFL tracking")
        ax.set_yticks(y, d["position_group"].to_list())
        for i, (n, r) in enumerate(zip(d["n_combine_athletes"], d["match_rate"])):
            ax.text(n, i, f"  {r:.0%} matched", va="center", fontsize=9, color=INK["secondary"])
        ax.set_xlabel("Athletes")
        ax.set_title("Coverage by position group")
        ax.legend(loc="lower right")
        ax.grid(axis="y", visible=False)
        add_caption(fig, "Position group = combine_position, else nfl_position. Matched = present in both tracking domains.")
        return save_figure(fig, self.cfg.paths.figures, "04_coverage_by_position")

    def plot_overlap(self, overlap: pl.DataFrame) -> Path:
        import numpy as np
        from matplotlib.colors import LinearSegmentedColormap

        names = list(ID_SOURCES)
        mat = np.array([[overlap.filter((pl.col("source_a") == a) & (pl.col("source_b") == b))["n_common"][0]
                         for b in names] for a in names], dtype=float)
        fig, ax = new_figure(figsize=(10.2, 7.2))
        im = ax.imshow(mat, cmap=LinearSegmentedColormap.from_list("b", SEQUENTIAL_BLUE))
        labels = [SHORT[n] for n in names]
        ax.set_xticks(range(len(names)), labels, rotation=30, ha="right")
        ax.set_yticks(range(len(names)), labels)
        ax.grid(False)
        vmax = mat.max() or 1
        for i in range(len(names)):
            for j in range(len(names)):
                ax.text(j, i, f"{int(mat[i, j]):,}", ha="center", va="center", fontsize=9,
                        color="white" if mat[i, j] > 0.6 * vmax else INK["primary"])
        fig.colorbar(im, ax=ax, label="Athletes in common", shrink=0.8)
        ax.set_title("nfl_id overlap between sources")
        add_caption(fig, "Diagonal = distinct athletes per source.")
        return save_figure(fig, self.cfg.paths.figures, "04_id_overlap_matrix")

    # ------------------------------------------------------------------ outputs
    def write_mermaid(self, card: pl.DataFrame) -> Path:
        lines = ["%% Source relationships — generated by nflbdb.data.relationships (measured)", "erDiagram"]
        for r in card.filter(~pl.col("purpose").str.starts_with("ANTI")).iter_rows(named=True):
            rel = {"1:1": "||--||", "N:1": "}o--||", "1:N": "||--o{", "N:M": "}o--o{"}[r["cardinality"]]
            lines.append(f'    {r["left"]} {rel} {r["right"]} : "{r["keys"]} ({r["left_match_rate"]:.1%} matched)"')
        return atomic_write_text("\n".join(lines) + "\n", self.cfg.paths.diagrams / "source_relationships.mmd")

    def run(self) -> dict[str, pl.DataFrame]:
        t = self.cfg.paths.tables
        presence = self.athlete_presence()
        res = {
            "namespace": self.id_namespace(),
            "overlap": self.overlap_matrix(),
            "presence": presence,
            "funnel": self.linkage_funnel(presence),
            "by_position": self.coverage_by(presence, "position_group"),
            "by_cohort": self.coverage_by(presence, "cohort"),
            "ambiguity": self.ambiguity_checks(),
            "cardinality": self.join_cardinality(),
        }
        stems = {"namespace": "rel_id_namespace", "overlap": "rel_id_overlap", "presence": "athlete_source_presence",
                 "funnel": "athlete_linkage_funnel", "by_position": "coverage_by_position",
                 "by_cohort": "coverage_by_cohort", "ambiguity": "identity_ambiguity", "cardinality": "join_cardinality"}
        for k, df in res.items():
            if not df.is_empty():
                export_table(df, t, stems[k])
        return res

    def write_report(self, res: dict[str, pl.DataFrame]) -> Path:
        p = self.cfg.paths
        figs = [self.plot_funnel(res["funnel"]), self.plot_coverage_by_position(res["by_position"]),
                self.plot_overlap(res["overlap"])]
        if not res["cardinality"].is_empty():
            self.write_mermaid(res["cardinality"])
        pres = res["presence"]
        rpt = MarkdownReport("04 — Relationship analysis", "Step 4: entity relationships, athlete linkage and join cardinality")
        rpt.h2("Athlete linkage summary").kv({
            "Combine athletes (results ∪ tracking)": pres.filter(pl.col("in_cmb_results") | pl.col("in_cmb_tracking")).height,
            "NFL tracked athletes": pres.filter(pl.col("in_nfl_tracking")).height,
            "Matched (Combine tracking ∧ NFL tracking)": pres.filter(pl.col("in_cmb_tracking") & pl.col("in_nfl_tracking")).height,
            "Combine athletes without NFL tracking": pres.filter((pl.col("in_cmb_results") | pl.col("in_cmb_tracking")) & ~pl.col("in_nfl_tracking")).height,
            "Identity ambiguities flagged": res["ambiguity"].height,
            "Linkage method": "exact nfl_id equality only",
        })
        for f in figs:
            if f:
                rpt.image(f, p.reports_root, "figure")
        rpt.h2("Linkage funnel").table(res["funnel"])
        rpt.h2("Coverage by position group").table(res["by_position"])
        rpt.h2("Coverage by cohort (draft year)").table(res["by_cohort"])
        rpt.h2("Identifier namespace check").table(res["namespace"])
        rpt.p("Overlapping value ranges and high intersection counts support (but do not prove) a shared nfl_id "
              "namespace. Disjoint ranges for a source would indicate a different namespace requiring a crosswalk.")
        rpt.h2("Identity ambiguity diagnostics")
        rpt.table(res["ambiguity"]) if res["ambiguity"].height else rpt.p("_No ambiguities detected._")
        rpt.h2("Join cardinality (estimated from per-key counts — no join materialised)")
        if not res["cardinality"].is_empty():
            rpt.table(res["cardinality"].select("left", "right", "keys", "cardinality", "n_left", "n_right",
                                                "est_inner_rows", "multiplicative_factor", "left_match_rate",
                                                "explosion_flag", "purpose"))
        rpt.callout("IMPORTANT", "Never join frame-level tables of the two domains directly. Aggregate Combine and NFL "
                    "tracking to athlete level (gold marts) and join those compact tables on nfl_id.")
        return rpt.save(p.reports_root / "04_relationship_analysis.md")
