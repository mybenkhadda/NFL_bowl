"""Read-only access to everything Milestone 11 reuses. Nothing of the warehouse is rebuilt; nothing upstream is written.

Combine
    ``gold/phase_03/combine_frame_kinematics``   frames of EVERY tracked drill (x_s, y_s, vx, vy, speed, a_tan, a_lat, …)
    ``gold/phase_03/combine_trajectory_quality`` quality tier per attempt (``reject`` attempts are excluded)
    ``athlete_registry`` (notebook 00)           position, draft year, official forty / 10-yd split / weight / height
    ``gold.mart_athlete_research_base``         official non-forty times (column names detected by substring)
    ``sequence_frames`` (notebook 00)            the eligible forty sequences (used for the sprint models)
NFL
    ``nfl_movement_demand_episodes`` (09A)       one row per physical episode of an eligible player-play
    ``nfl_play_movement`` (05A)                  eligible player-plays, context, broad play measures (``accel_sust05``)
Shared
    ``transfer_outer_folds`` (06A)               athlete-grouped folds reused when present
    ``stopwatch_twins``, ``combine_curve_prospect_features`` (09B / 09C)

Drill names are read from the warehouse, never assumed: ``DRILL_FAMILIES`` maps them to physical families by pattern,
and the notebook shows the mapping (with ``OVERRIDES``) before anything is computed.
"""

from __future__ import annotations

import re
from typing import Any, Sequence

import polars as pl

from ....data.utils import get_logger
from .. import nfl_outcomes as no
from ..config import AdvancedConfig
from ..data import KEYS, OFFICIAL, AdvInputs
from .config import M11Config

log = get_logger(__name__)
NB = {"00": "00_research_setup_and_data_audit.ipynb", "05A": "05a_nfl_movement_outcome_eda.ipynb", "06A": "06a_transfer_cohort_and_leakage_audit.ipynb",
      "09A": "09a_nfl_movement_demand_atlas.ipynb", "09B": "09b_cross_domain_movement_matching.ipynb", "09C": "09c_stopwatch_twins_discovery.ipynb",
      "11.0": "11_00_research_setup_and_opportunity_inventory.ipynb", "11A": "11a_drill_to_nfl_transfer_matrix.ipynb",
      "11B": "11b_speed_state_transfer_and_envelopes.ipynb", "11C": "11c_braking_turning_reacceleration.ipynb",
      "11D": "11d_nfl_athletic_development.ipynb", "11E": "11e_stopwatch_twin_discordance.ipynb", "11F": "11f_shared_combine_nfl_representation.ipynb",
      "11G": "11g_optimal_combine_drill_selection.ipynb", "11H": "11h_position_specific_football_actions.ipynb",
      "11I": "11i_novel_metric_tournament_and_synthesis.ipynb"}

# physical drill families, matched in order on the UPPER-CASE drill name (first match wins)
DRILL_FAMILIES: tuple[tuple[str, str, str], ...] = (
    ("forty", r"FORTY|40_?Y(AR)?D|^40", "straight sprint from a stance; the timed 40 yd"),
    ("long_shuttle", r"60_?Y(AR)?D|LONG_SHUTTLE", "repeated 180° reversals over longer legs"),
    ("short_shuttle", r"SHUTTLE|PRO_?AGILITY|5_?10_?5|20_?Y(AR)?D", "lateral start, two 180° reversals through near-stops"),
    ("three_cone", r"CONE|L_?DRILL", "90° and 180° turns, curved running around cones"),
    ("gauntlet", r"GAUNTLET", "lateral run at speed across the field (receivers)"),
    ("backpedal_transition", r"BACK_?PEDAL|W_?DRILL|T_?STEP|SPEED_?TURN|WEAVE|BREAK|TRANSITION|OPEN_?HIP|HIP_?TURN|COD",
     "backward / lateral movement with a transition into forward acceleration (defensive backs)"),
    ("pass_rush", r"RUSH|BAG|HOOP|RIP|DIP|SWIM|GET_?OFF|STANCE|PUNCH|MIRROR|PULL|KICK|SLED|CHUTE|SET",
     "short launches from a stance, lateral steps, repeated short accelerations (linemen)"),
)
FAMILY_ORDER = tuple(f for f, _, _ in DRILL_FAMILIES) + ("other",)
FRAME_COLUMNS = (*KEYS, "drill", "drill_name", "drill_type", "draft_year", "seg_idx", "ts", "t_s", "valid_kin", "x_s", "y_s", "vx", "vy", "ax", "ay",
                 "speed", "a_tan", "a_lat", "yaw_rate", "heading_valid", "step_smooth", "pos", "seg_len", "boundary")


def drill_family(name: str | None, overrides: dict[str, str] | None = None) -> str:
    if name is None:
        return "other"
    if overrides and name in overrides:
        return overrides[name]
    up = re.sub(r"[\s\-]+", "_", str(name).upper())
    return next((fam for fam, pat, _ in DRILL_FAMILIES if re.search(pat, up)), "other")


class M11Sources:
    """``store``: ``artifacts.DiscoveryStore``. ``athlete_ids``: optional subset (smoke)."""

    def __init__(self, cfg: Any, store: Any, c: M11Config, athlete_ids: Sequence[int] | None = None, overrides: dict[str, str] | None = None):
        self.cfg, self.store, self.c = cfg, store, c
        self.acfg = AdvancedConfig().for_mode(store.controls)
        self.inputs = AdvInputs(cfg, self.acfg, athlete_ids)
        self.overrides = overrides or {}
        self.oc = no.OutcomeConfig().for_mode(store.controls)

    # ------------------------------------------------------------------ Combine
    def frame_schema(self) -> list[str]:
        return self.inputs.frames().collect_schema().names()

    def drill_names(self) -> pl.DataFrame:
        """Distinct drill identifiers in the frames, with their family under the current mapping."""
        cols = [k for k in ("drill", "drill_name", "drill_type") if k in self.frame_schema()]
        d = (self.inputs.frames().group_by(cols).agg(pl.struct(KEYS).n_unique().alias("n_attempts"), pl.col("nfl_id").n_unique().alias("n_athletes"),
                                                       pl.len().alias("n_frames")).collect().sort("drill"))
        return d.with_columns(pl.col("drill").map_elements(lambda s: drill_family(s, self.overrides), return_dtype=pl.String).alias("family"))

    def combine_frames(self, families: Sequence[str] | None = None, usable_only: bool = True) -> pl.DataFrame:
        """Frames of the requested drill families (all when None), in time order, with ``family`` and ``seq_id``."""
        have = self.frame_schema()
        lf = self.inputs.frames().select([k for k in FRAME_COLUMNS if k in have])
        names = self.drill_names()
        if families is not None:
            keep = names.filter(pl.col("family").is_in(list(families)))["drill"].to_list()
            lf = lf.filter(pl.col("drill").is_in(keep))
        df = lf.collect()
        if usable_only:
            q = self.inputs.trajectory_quality()
            if q is not None:
                df = df.join(q.filter(pl.col("quality_tier") != "reject").select(KEYS), on=KEYS, how="semi")
        fam = dict(zip(names["drill"].to_list(), names["family"].to_list()))
        df = df.with_columns(pl.col("drill").replace_strict(fam, default="other", return_dtype=pl.String).alias("family"),
                             (pl.col("event_id").cast(pl.String) + "|" + pl.col("nfl_id").cast(pl.String) + "|" + pl.col("attempt").cast(pl.String)).alias("seq_id"))
        return df.sort("seq_id", "seg_idx", "ts")

    def athletes(self) -> pl.DataFrame:
        return self.store.upstream.require_named("athlete_registry", NB["00"])

    def officials(self) -> pl.DataFrame:
        """(nfl_id, position_group, draft_year, forty, ten, weight, height, three_cone, short_shuttle) — official values only."""
        a = self.athletes().select("nfl_id", "position_group", "draft_year", *[pl.col(k).alias(k.removeprefix("cmb_").replace("ten_yd_split", "ten")
                                                                                              .replace("combine_", "")) for k in OFFICIAL])
        try:
            m = self.inputs.mart()
            extra = {}
            for want, pats in (("three_cone", ("three_cone", "3cone", "cone")), ("short_shuttle", ("short_shuttle", "shuttle"))):
                col = next((k for p in pats for k in sorted(m.columns) if p in k.lower() and m.schema[k].is_numeric()), None)
                if col:
                    extra[want] = col
            if extra:
                a = a.join(m.select("nfl_id", *[pl.col(v).cast(pl.Float64).alias(k) for k, v in extra.items()]).unique(subset="nfl_id"), on="nfl_id", how="left")
            log.info("[m11.sources] official non-forty columns: %s", extra or "none found")
        except Exception as e:                                         # noqa: BLE001 — the mart is optional for officials
            log.warning("[m11.sources] mart_athlete_research_base unreadable (%s): no official non-forty times", e)
        for k in ("three_cone", "short_shuttle"):
            if k not in a.columns:
                a = a.with_columns(pl.lit(None, dtype=pl.Float64).alias(k))
        return a

    def sequence_frames(self) -> pl.DataFrame:
        return self.store.upstream.require_named("sequence_frames", NB["00"])

    def phase03(self, name: str) -> pl.DataFrame | None:
        p03 = self.inputs.p03
        return self.inputs._restrict(p03.load(name).collect()) if p03.dataset_files(name) else None

    # ------------------------------------------------------------------ NFL
    def episodes(self) -> pl.DataFrame:
        ep = self.store.intelligence.require_named("nfl_movement_demand_episodes", NB["09A"])
        return ep.filter(~pl.col("edge"))

    def plays(self) -> pl.DataFrame:
        return self.store.nfl.require(no.plays_contract(self.oc), NB["05A"])

    # ------------------------------------------------------------------ shared
    def transfer_folds(self) -> pl.DataFrame | None:
        try:
            return self.store.transfer.require_named("transfer_outer_folds", NB["06A"])
        except FileNotFoundError:
            return None

    def twins(self) -> pl.DataFrame:
        return self.store.intelligence.require_named("stopwatch_twins", NB["09C"])

    def curve_prospects(self) -> pl.DataFrame:
        return self.store.intelligence.require_named("combine_curve_prospect_features", NB["09B"])

    def demand_registry(self) -> pl.DataFrame:
        return self.store.intelligence.require_named("nfl_movement_demand_registry", NB["09A"])

    def input_files(self) -> list[Any]:
        return self.inputs.input_files()


__all__ = ["DRILL_FAMILIES", "FAMILY_ORDER", "FRAME_COLUMNS", "M11Sources", "NB", "drill_family"]
