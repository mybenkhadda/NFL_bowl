"""Research access layer — the ONLY entry point later notebooks should need.

All methods return Polars ``LazyFrame``s (filters and column selections are pushed down to
the Parquet scan; nothing is read until ``.collect()``), except :meth:`sql`, which runs DuckDB
against the catalog views and returns a Polars DataFrame.

Typical use::

    from nflbdb.data.config import Phase01Config, configure_runtime
    cfg = Phase01Config.load(); configure_runtime(cfg)
    from nflbdb.data.access import ResearchDataAccess
    rda = ResearchDataAccess(cfg)

    wr = rda.athletes(position="WR").collect()
    frames = rda.nfl_frames(game_id=2024010707, play_id=570).collect()          # one play, row-group pruned
    traj = rda.sample_trajectories("combine", n=20, drill_name="OL_PULL_DRILL_FOLD_BLOCK_RIGHT").collect()
    base = rda.research_base(require=("combine_tracking", "nfl_tracking")).collect()
    rda.sql("SELECT position_group, count(*) FROM gold_dim_athlete GROUP BY 1")
"""

from __future__ import annotations

from typing import Iterable, Sequence

import polars as pl

from .config import Phase01Config
from .frames import query_df
from .tables import LayerRefs


def _as_list(v) -> list | None:
    if v is None:
        return None
    return list(v) if isinstance(v, (list, tuple, set)) else [v]


class ResearchDataAccess:
    def __init__(self, cfg: Phase01Config):
        self.cfg = cfg
        self.refs = LayerRefs(cfg)
        self._con = None

    # ------------------------------------------------------------------ raw handles
    def table(self, name: str, layer: str = "gold") -> pl.LazyFrame:
        return self.refs.scan(layer, name)

    def sql(self, query: str) -> pl.DataFrame:
        """DuckDB over catalog views (``gold_*``, ``silver_*``, ``fact_nfl_tracking`` …)."""
        if self._con is None:
            from .catalog import WarehouseCatalog

            self._con = WarehouseCatalog(self.cfg).connect(read_only=True)
        return query_df(self._con, query)

    # ------------------------------------------------------------------ athletes
    def athletes(self, position: str | Sequence[str] | None = None, cohort: int | Sequence[int] | None = None,
                 require: Iterable[str] = ()) -> pl.LazyFrame:
        """``require`` ⊆ {combine_results, combine_tracking, nfl_tracking, player_play, career}."""
        lf = self.table("dim_athlete")
        if (pos := _as_list(position)) is not None:
            lf = lf.filter(pl.col("position_group").is_in(pos) | pl.col("nfl_position").is_in(pos))
        if (coh := _as_list(cohort)) is not None:
            lf = lf.filter(pl.col("cohort").is_in(coh))
        for r in require:
            lf = lf.filter(pl.col(f"has_{r}"))
        return lf

    def coverage(self) -> pl.LazyFrame:
        return self.table("mart_combine_nfl_coverage")

    def research_base(self, position: str | Sequence[str] | None = None, require: Iterable[str] = ()) -> pl.LazyFrame:
        """Compact athlete-level table. ``career_*`` columns are outcomes, never predictors."""
        lf = self.table("mart_athlete_research_base")
        if (pos := _as_list(position)) is not None:
            lf = lf.filter(pl.col("position_group").is_in(pos))
        for r in require:
            lf = lf.filter(pl.col(f"has_{r}"))
        return lf

    # ------------------------------------------------------------------ combine
    def combine_results(self, nfl_id: int | Sequence[int] | None = None) -> pl.LazyFrame:
        lf = self.table("combine_results", "silver")
        if (ids := _as_list(nfl_id)) is not None:
            lf = lf.filter(pl.col("nfl_id").is_in(ids))
        return lf

    def combine_attempts(self, nfl_id=None, drill_name=None, draft_year=None) -> pl.LazyFrame:
        lf = self.table("fact_combine_attempt")
        for col, v in (("nfl_id", nfl_id), ("drill_name", drill_name), ("draft_year", draft_year)):
            if (vals := _as_list(v)) is not None:
                lf = lf.filter(pl.col(col).is_in(vals))
        return lf

    def combine_frames(self, nfl_id=None, drill_name=None, event_id=None, attempt=None, draft_year=None,
                       columns: Sequence[str] | None = None) -> pl.LazyFrame:
        """Frame-level Combine tracking. ``draft_year`` prunes Hive partitions; ``nfl_id`` and
        ``drill_name`` prune row groups (silver is sorted by nfl_id, drill_name)."""
        lf = self.table("combine_tracking", "silver")
        for col, v in (("draft_year", draft_year), ("nfl_id", nfl_id), ("drill_name", drill_name),
                       ("event_id", event_id), ("attempt", attempt)):
            if (vals := _as_list(v)) is not None:
                lf = lf.filter(pl.col(col).is_in(vals))
        return lf.select(columns) if columns else lf

    # ------------------------------------------------------------------ NFL
    def nfl_plays_for_athlete(self, nfl_id: int | Sequence[int], season=None) -> pl.LazyFrame:
        """Player×play summaries (one row per tracked play) joined to play context."""
        lf = self.table("fact_nfl_play_participation").filter(pl.col("nfl_id").is_in(_as_list(nfl_id)))
        if (s := _as_list(season)) is not None:
            lf = lf.filter(pl.col("season").is_in(s))
        plays = self.table("dim_play")
        fact_cols = set(lf.collect_schema().names())
        overlap = [c for c in plays.collect_schema().names() if c in fact_cols and c not in ("game_id", "play_id")]
        return lf.join(plays.drop(overlap), on=["game_id", "play_id"], how="left")

    def nfl_frames(self, game_id=None, play_id=None, nfl_id=None, season=None, include_football: bool = False,
                   columns: Sequence[str] | None = None) -> pl.LazyFrame:
        """Frame-level game tracking. Always filter by ``season`` and/or ``game_id`` for speed:
        ``season`` prunes partitions; ``game_id``/``play_id`` prune row groups (sorted)."""
        lf = self.table("game_tracking", "silver")
        for col, v in (("season", season), ("game_id", game_id), ("play_id", play_id), ("nfl_id", nfl_id)):
            if (vals := _as_list(v)) is not None:
                lf = lf.filter(pl.col(col).is_in(vals))
        if not include_football:
            lf = lf.filter(~pl.col("is_football"))
        return lf.select(columns) if columns else lf

    def athlete_nfl_profile(self, nfl_id=None, season=None) -> pl.LazyFrame:
        lf = self.table("mart_athlete_nfl_profile")
        for col, v in (("nfl_id", nfl_id), ("season", season)):
            if (vals := _as_list(v)) is not None:
                lf = lf.filter(pl.col(col).is_in(vals))
        return lf

    # ------------------------------------------------------------------ samples
    def sample_trajectories(self, domain: str = "nfl", n: int = 20, seed: int | None = None, **filters) -> pl.LazyFrame:
        """Frames of ``n`` randomly chosen trajectories (deterministic for a seed). Keys are
        sampled from the compact fact table first, then only those frames are scanned."""
        seed = self.cfg.profiling.seed if seed is None else seed
        if domain == "nfl":
            keys = ["game_id", "play_id", "nfl_id"]
            facts = self.table("fact_nfl_play_participation")
            frames = self.table("game_tracking", "silver").filter(~pl.col("is_football"))
        elif domain == "combine":
            keys = ["event_id", "nfl_id", "attempt"]
            facts = self.table("fact_combine_attempt")
            frames = self.table("combine_tracking", "silver")
        else:
            raise ValueError("domain must be 'nfl' or 'combine'")
        for col, v in filters.items():
            facts = facts.filter(pl.col(col).is_in(_as_list(v)))
        chosen = facts.select(keys + [c for c in ("season", "draft_year") if c in facts.collect_schema().names()])
        chosen = chosen.collect().sample(n=n, seed=seed, shuffle=True) if n else chosen.collect()
        if domain == "nfl":  # partition + row-group pruning on the frame scan
            frames = frames.filter(pl.col("season").is_in(chosen["season"].unique().to_list())
                                   & pl.col("game_id").is_in(chosen["game_id"].unique().to_list()))
        else:
            frames = frames.filter(pl.col("draft_year").is_in(chosen["draft_year"].unique().to_list())
                                   & pl.col("nfl_id").is_in(chosen["nfl_id"].unique().to_list()))
        sort_col = "ts"
        return frames.join(chosen.lazy().select(keys), on=keys, how="semi").sort(keys + [sort_col])
