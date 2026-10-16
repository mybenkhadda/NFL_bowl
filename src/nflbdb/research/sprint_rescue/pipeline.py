"""Inputs and stage orchestration of the sprint-rescue experiment (reads persisted datasets; rebuilds nothing).

Stage order and dependencies::

    ATT  attempt table, model refits, window distribution, within-athlete differences, example signals
    A    censoring association            ← ATT
    B    within-position reliability      ← ATT
    C    model repeatability              ← ATT
    D    athlete table, redundancy, screening   ← ATT, B
    E    cohort reconciliation            ← ATT
    F    NFL endpoint feasibility         ← ATT

A stage is reused when its fingerprint (parameters, upstream files' size + mtime, fingerprints of the stages it
depends on) is unchanged and all its outputs exist.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import polars as pl

from ...data.tables import LayerRefs
from ...data.utils import file_fingerprint, get_logger
from ...features.config import KinematicsConfig, MovementConfig
from ..artifacts import ArtifactStore
from ..contracts import TableContract, cast_keys
from . import attempts as at
from . import censoring as ce
from . import cohorts as co
from . import redundancy as rd
from . import reliability as rl
from .config import RescueConfig, feature_specs
from .sprint_model import FITS
from .store import RescueStore

log = get_logger(__name__)
P03_DATASETS = ("combine_sprint_profiles", "combine_attempt_features", "combine_frame_kinematics")
P04_DATASETS = ("trajectory_features_by_smoothing", "trajectory_quality_flags")
STAGES: dict[str, tuple[tuple[TableContract, ...], tuple[str, ...]]] = {
    "ATT": ((at.ATTEMPTS, FITS, at.WINDOWS, at.WITHIN, at.EXAMPLES), ()),
    "A": ((ce.ASSOC,), ("ATT",)),
    "B": ((rl.RELIABILITY,), ("ATT",)),
    "C": ((rl.MODEL_REPEAT,), ("ATT",)),
    "D": ((rd.ATHLETE, rd.REDUNDANCY, rd.SCREEN), ("ATT", "B")),
    "E": ((co.RECON,), ("ATT",)),
    "F": ((co.ENDPOINT,), ("ATT",)),
}
ALL_CONTRACTS = tuple(c for contracts, _ in STAGES.values() for c in contracts)


class RescueInputs:
    """Read-only access to the persisted upstream datasets."""

    def __init__(self, cfg: Any, rcfg: RescueConfig, sample_athletes: int | None = None):
        self.cfg, self.rcfg, self.sample_athletes = cfg, rcfg, sample_athletes
        self.refs = LayerRefs(cfg)
        self.p03 = ArtifactStore(cfg, phase="phase_03")
        self.mcfg = MovementConfig()
        primary = KinematicsConfig().smoothing
        if primary.label != rcfg.smoothing:
            raise ValueError(f"RescueConfig.smoothing {rcfg.smoothing!r} is not the primary SG setting {primary.label!r} "
                             "that the persisted Phase 03 frames were built with")
        self.half = primary.window // 2
        self._ids: list[int] | None = None

    # ------------------------------------------------------------------ files
    def p04_path(self, name: str) -> Path:
        return self.cfg.paths.gold / "phase_04" / "quality" / f"{name}.parquet"

    def upstream_files(self, stage: str) -> list[Path]:
        if stage == "ATT":
            files = [f for n in P03_DATASETS for f in self.p03.dataset_files(n)]
            return [*files, *[self.p04_path(n) for n in P04_DATASETS], *self.refs.files("gold", "mart_athlete_research_base")]
        if stage == "D":
            return [self.registry_path()]
        if stage in ("E", "F"):
            return [f for layer, n in (("gold", "fact_nfl_play_participation"), ("silver", "player_play"), ("silver", "plays"),
                                       ("silver", "games"), ("gold", "dim_athlete"), ("silver", "combine_results"),
                                       ("gold", "mart_athlete_research_base")) for f in self.refs.files(layer, n)]
        return []

    def upstream_state(self, stage: str) -> dict[str, Any]:
        return {self.cfg.paths.rel(f): (file_fingerprint(f) if f.is_file() else "missing") for f in self.upstream_files(stage)}

    def registry_path(self) -> Path:
        return self.cfg.paths.tables / "phase_04" / "reliability" / "V_feature_validation_registry.csv"

    # ------------------------------------------------------------------ Combine side
    def _sample(self, df: pl.DataFrame) -> pl.DataFrame:
        if self.sample_athletes is None:
            return df
        if self._ids is None:
            self._ids = sorted(self.sprint_all()["nfl_id"].unique().to_list())[: self.sample_athletes]
        return df.filter(pl.col("nfl_id").is_in(self._ids))

    def sprint_all(self) -> pl.DataFrame:
        return cast_keys(self.p03.load("combine_sprint_profiles").collect())

    def sprint(self) -> pl.DataFrame:
        return self._sample(self.sprint_all())

    def attempt_features(self) -> pl.DataFrame:
        lf = self.p03.load("combine_attempt_features").filter(pl.col("drill") == self.rcfg.drill)
        return self._sample(cast_keys(lf.select("event_id", "nfl_id", "attempt", "accel_sustained", "accel_initial",
                                                "resid_rms_yds").collect()))

    def frames(self) -> pl.DataFrame:
        lf = self.p03.load("combine_frame_kinematics").filter(pl.col("drill") == self.rcfg.drill)
        return self._sample(cast_keys(lf.select(at.FRAME_COLUMNS).collect()))

    def p04(self, name: str) -> pl.DataFrame:
        p = self.p04_path(name)
        if not p.is_file():
            raise FileNotFoundError(f"{self.cfg.paths.rel(p)} missing — run `python -m nflbdb.research.phase04 --only Q,B`")
        return pl.read_parquet(p)

    def research_base(self) -> pl.DataFrame:
        lf = self.refs.scan("gold", "mart_athlete_research_base")
        want = ["nfl_id", *self.rcfg.official, *self.rcfg.traditional]
        return cast_keys(lf.select([c for c in want if c in lf.collect_schema().names()]).collect())

    def registry(self) -> dict[str, str]:
        p = self.registry_path()
        if not p.is_file():
            return {}
        r = pl.read_csv(p, infer_schema_length=0).filter((pl.col("domain") == "combine") & (pl.col("drill") == self.rcfg.drill)
                                                         & (pl.col("stratum") == self.rcfg.drill))
        return dict(r.select("feature", "status").iter_rows())

    # ------------------------------------------------------------------ NFL side
    def nfl_base(self) -> pl.DataFrame:
        """Tracked player-plays of every season type, assembled exactly as the Phase 05 P0 census does."""
        from ..phase05 import decision_audit as da
        from ..phase05.config import Phase05Config

        aud = da.DecisionAuditStudy(self.cfg, Phase05Config())
        aud.roles = {"state": tuple(self.rcfg.context_fields), "action": (),
                     "outcome": tuple(e for e, _ in self.rcfg.endpoints)}
        inp = aud.inputs()
        fields = [f for r in da.ROLES for f in aud.roles[r]]
        base, _ = da.assemble_base(inp["fact"], inp["player_play"], inp["plays"], inp["games"], inp["athletes"], fields)
        return base

    def combine_ids(self) -> list[int]:
        return self.refs.scan("silver", "combine_results").select(pl.col("nfl_id").cast(pl.Int64)).unique().collect()[
            "nfl_id"].drop_nulls().to_list()

    def definitions(self) -> dict[str, dict[str, str]]:
        p = self.cfg.paths.tables / "schema_catalog.csv"
        if not p.is_file():
            return {}
        cat = pl.read_csv(p, infer_schema_length=0).filter(pl.col("table") == "player_play")
        return {r["column"]: {"meaning": r["meaning"] or "", "unit": r["unit"] or "", "evidence": r["evidence"] or ""}
                for r in cat.iter_rows(named=True)}


# ============================================================================== stage computations
def compute_att(inp: RescueInputs, store: RescueStore) -> dict[str, pl.DataFrame]:
    rcfg = inp.rcfg
    frames = inp.frames()
    frame_feats, fits = at.frame_products(frames, inp.mcfg, rcfg, inp.half)
    att = at.build_attempt_table(inp.sprint(), inp.attempt_features(), inp.p04(P04_DATASETS[0]),
                                 inp.p04(P04_DATASETS[1]), frame_feats, fits, inp.research_base(), rcfg)
    unmatched = att.filter(pl.col("p95_a_tan").is_null()).height
    if unmatched:
        log.warning("[sprint_rescue:ATT] %s of %s forty attempts have no Phase 04 feature row (kept, values null)",
                    f"{unmatched:,}", f"{att.height:,}")
    return {at.ATTEMPTS.name: att, FITS.name: fits, at.WINDOWS.name: at.window_distribution(att),
            at.WITHIN.name: at.within_athlete_variation(att),
            at.EXAMPLES.name: at.example_signals(frames, fits, at.pick_examples(att), rcfg)}


def compute_a(inp: RescueInputs, store: RescueStore) -> dict[str, pl.DataFrame]:
    return {ce.ASSOC.name: ce.association_table(store.read(at.ATTEMPTS), store.read(at.WITHIN), inp.rcfg)}


def compute_b(inp: RescueInputs, store: RescueStore) -> dict[str, pl.DataFrame]:
    return {rl.RELIABILITY.name: rl.reliability_table(store.read(at.ATTEMPTS), feature_specs(inp.rcfg), inp.rcfg)}


def compute_c(inp: RescueInputs, store: RescueStore) -> dict[str, pl.DataFrame]:
    return {rl.MODEL_REPEAT.name: rl.model_repeatability(store.read(at.ATTEMPTS), store.read(FITS), inp.rcfg)}


def compute_d(inp: RescueInputs, store: RescueStore) -> dict[str, pl.DataFrame]:
    specs = feature_specs(inp.rcfg)
    athletes = rd.athlete_table(store.read(at.ATTEMPTS), inp.rcfg)
    red = rd.redundancy_table(athletes, specs, inp.rcfg)
    return {rd.ATHLETE.name: athletes, rd.REDUNDANCY.name: red,
            rd.SCREEN.name: rd.screening_summary(specs, store.read(rl.RELIABILITY), red, inp.registry(), inp.rcfg)}


def nfl_season_types(base: pl.DataFrame) -> pl.DataFrame:
    return base.group_by("nfl_id", "season_type").agg(pl.len().cast(pl.Int64).alias("n_plays"))


def compute_e(inp: RescueInputs, store: RescueStore) -> dict[str, pl.DataFrame]:
    return {co.RECON.name: co.cohort_reconciliation(inp.combine_ids(), nfl_season_types(inp.nfl_base()),
                                                    store.read(at.ATTEMPTS), inp.research_base(), inp.rcfg)}


def compute_f(inp: RescueInputs, store: RescueStore) -> dict[str, pl.DataFrame]:
    pop = inp.nfl_base().filter(pl.col("season_type").is_in(list(inp.rcfg.season_types)))
    flags = co.combine_flags(store.read(at.ATTEMPTS), inp.research_base(), inp.rcfg)
    return {co.ENDPOINT.name: co.endpoint_table(pop, flags, inp.definitions(), inp.rcfg)}


COMPUTE: dict[str, Callable[[RescueInputs, RescueStore], dict[str, pl.DataFrame]]] = {
    "ATT": compute_att, "A": compute_a, "B": compute_b, "C": compute_c, "D": compute_d, "E": compute_e, "F": compute_f}


def stage_fingerprint(stage: str, inp: RescueInputs, store: RescueStore) -> str:
    deps = {d: store.recorded(d) for d in STAGES[stage][1]}
    params = {"rescue": inp.rcfg.params(), "movement": inp.mcfg.params(), "sample_athletes": inp.sample_athletes}
    return store.fingerprint(stage, params, {"files": inp.upstream_state(stage), "stages": deps})


def run_stage(stage: str, inp: RescueInputs, store: RescueStore, force: bool = False) -> None:
    contracts, deps = STAGES[stage]
    missing = [d for d in deps if store.recorded(d) is None]
    if missing:
        raise RuntimeError(f"stage {stage} needs completed stage(s) {missing}: run "
                           f"`python -m nflbdb.research.sprint_rescue --only {','.join([*missing, stage])}`")
    fp = stage_fingerprint(stage, inp, store)
    current, reason = store.is_current(stage, fp, contracts)
    if current and not force:
        log.info("[sprint_rescue:%s] reuse (%s)", stage, reason)
        return
    log.info("[sprint_rescue:%s] compute (%s%s)", stage, "forced; " if force else "", reason)
    out = COMPUTE[stage](inp, store)
    provenance = {"inputs": list(inp.upstream_state(stage)), "depends_on": list(deps), "config": inp.rcfg.params()}
    for c in contracts:
        store.write(out[c.name], c, stage, provenance)
    store.record(stage, fp, contracts)      # recorded only after EVERY output passed its contract
