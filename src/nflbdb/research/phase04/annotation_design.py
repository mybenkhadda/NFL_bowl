"""Study C2 v2 — reference event annotation (Combine + NFL) and detector validation (Phase 04 · M1 · Study 1.3).

Extends the Phase 03 C2 framework (``studies/annotation.py``: blind packets, label schema, one-to-one
matcher, error taxonomy, P/R/F1 with athlete-cluster CIs) — nothing there is duplicated. New here:

* strata: forty, short shuttle, three-cone, WR / DB / DL / OL / TE position drills, and NFL player-plays by
  position group (deterministic NFL sample of Study A3);
* two selections per stratum, never pooled for the headline: ``random`` (seeded hash order → unbiased
  within-stratum precision / recall) and ``difficult`` (highest interior jerk → stress test). The previous
  C2 selection ('typical' by event-count sensitivity) used the detector itself to choose tasks;
  v2 selection is detector-blind;
* athlete-level split: ``dev`` (may be used to revise the detector) vs ``eval`` (frozen; reported).
  No athlete appears in both. Revised detectors are scored ONLY on ``eval``;
* ``priority`` = labelling order (eval-random first), so a partially labelled set still yields the headline;
* detections are recomputed for every SG setting × rule on the task trajectories, so detector accuracy can
  be compared across smoothing (not only rules); tolerance sensitivity (0.2 / 0.3 / 0.5 s).

Onset / offset definitions and the labelling procedure: ``docs/PHASE_04_ANNOTATION_PROTOCOL.md``.
No detector is declared validated without labels — absent labels, every event feature stays ≤ PROVISIONAL
(Study V).

Commands for the USER (code authored, NOT executed)::

    python -m nflbdb.research.phase04 --annotation-tasks          # tasks + blind packets + label template
    # label reports/annotation/phase_04/annotation_labels.csv (copy of the template) following the protocol
    python -m nflbdb.research.phase04 --evaluate-annotations reports/annotation/phase_04/annotation_labels.csv
"""

from __future__ import annotations

import dataclasses
import zlib
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from ...data.config import Phase01Config
from ...data.utils import get_logger
from ...features.config import KinematicsConfig
from ...features.events import detect_events
from ...features.kinematics import compute_kinematics
from ...features.trajectory import COMBINE, NFL, TrajectoryDomain, prepare_frames
from ..contracts import TableContract, cast_keys
from ..kinematic_build import KinematicFeatureBuilder
from ..studies import annotation as ann
from ..studies.quality import traj_id_expr
from ..studies.signal_reliability import SIGNALS
from ..studies.store import FigureRecord, StudyStore
from .config import AnnotationDesignConfig, Phase04Config

log = get_logger(__name__)
STUDY = "C2"
S, I, F, B = pl.String(), pl.Int64(), pl.Float64(), pl.Boolean()
ANNOTATION_DIR = Path("reports") / "annotation" / "phase_04"
CKEYS, NKEYS = ["event_id", "nfl_id", "attempt"], ["game_id", "play_id", "nfl_id"]

TASKS = TableContract(
    "C2_annotation_tasks", "Detector-blind, stratified annotation tasks with an athlete-level dev / eval split.",
    {"task_id": S, "domain": S, "traj_id": S, "nfl_id": I, "stratum": S, "position": S, "cohort": I,
     "selection": S, "split": S, "priority": I, "selection_rule": S, "duration_s": F},
    key=("task_id",), area="hypothesis_screening",
    docs={"selection": "random = seeded hash order (unbiased within stratum); difficult = highest interior jerk",
          "split": "dev = may inform a detector revision; eval = frozen, reported", "priority": "labelling order"})
MATCHES = TableContract(
    "C2_annotation_matches", "Reference ↔ detector matches per SG setting × rule × tolerance (error taxonomy).",
    {**ann.MATCHES.columns, "smoothing": S, "detector_rule": S, "tolerance_s": F, "split": S, "selection": S,
     "domain": S}, area="hypothesis_screening")
PERFORMANCE = TableContract(
    "C2_detector_performance", "Detector precision / recall / F1 / timing error vs human labels.",
    {"split": S, "selection_set": S, "smoothing": S, "detector_rule": S, "tolerance_s": F, "event_class": S,
     "facet": S, "level": S, "label_set": S, "n_tasks": I, "n_athletes": I, "tp": I, "fp": I, "fn": I, "precision": F,
     "precision_lower": F, "precision_upper": F, "recall": F, "recall_lower": F, "recall_upper": F, "f1": F,
     "f1_lower": F, "f1_upper": F, "onset_mae_s": F, "onset_median_s": F, "offset_mae_s": F, "count_mae": F},
    key=("split", "selection_set", "smoothing", "detector_rule", "tolerance_s", "event_class", "facet", "level",
         "label_set"), area="hypothesis_screening",
    docs={"selection_set": "random (headline), difficult (stress test) or all",
          "label_set": "all_labels or certain_only (uncertain reference events excluded)"})
TAXONOMY = TableContract(
    "C2_error_taxonomy", "Counts of TP / FP / FN by error type (primary tolerance).",
    {"split": S, "smoothing": S, "detector_rule": S, "event_class": S, "outcome": S, "error_type": S, "n": I},
    key=("split", "smoothing", "detector_rule", "event_class", "outcome", "error_type"), area="hypothesis_screening")
COVERAGE = TableContract(
    "C2_label_coverage", "How many tasks have labels, by split × selection × stratum.",
    {"split": S, "selection": S, "stratum": S, "n_tasks": I, "n_labelled": I, "share_labelled": F},
    key=("split", "selection", "stratum"), area="hypothesis_screening")


def _h(seed: int, *parts: Any) -> int:
    return zlib.crc32(f"{seed}|{'|'.join(map(str, parts))}".encode())


# ============================================================================== 1. task selection (pure)
def assign_split(nfl_ids: list[int], dev_fraction: float, seed: int) -> dict[int, str]:
    return {i: ("dev" if _h(seed, "split", i) % 10_000 < dev_fraction * 10_000 else "eval") for i in nfl_ids}


def select_tasks(signals: pl.DataFrame, cfg: AnnotationDesignConfig, seed: int) -> pl.DataFrame:
    """``signals``: Study A3 ``trajectory_signal_reliability`` (Combine + nfl_sample). Detector-blind."""
    c = signals.filter((pl.col("domain") == "combine") & pl.col("stratum").is_in(list(cfg.combine_strata)))
    n = (signals.filter((pl.col("domain") == "nfl_sample") & pl.col("position").is_in(list(cfg.nfl_positions)))
         .with_columns(pl.format("nfl_play:{}", pl.col("position")).alias("stratum")))
    pool = (pl.concat([c, n]).filter((pl.col("quality_tier") != "reject") & (pl.col("n_valid_kin") > 0))
            .select("domain", "traj_id", "nfl_id", "stratum", "position", "cohort", "duration_s", "jerk_rms_interior")
            .with_columns(pl.col("traj_id").map_elements(lambda t: _h(seed, "task", t), return_dtype=pl.Int64)
                          .alias("_hash")))
    parts = []
    for (stratum,), g in pool.group_by("stratum", maintain_order=True):
        rnd = g.sort("_hash", "traj_id").head(cfg.per_stratum_random)
        dif = (g.filter(~pl.col("traj_id").is_in(rnd["traj_id"].to_list()) & pl.col("jerk_rms_interior").is_not_null())
               .sort(["jerk_rms_interior", "traj_id"], descending=[True, False]).head(cfg.per_stratum_difficult))
        parts += [rnd.with_columns(pl.lit("random").alias("selection"),
                                   pl.lit("seeded hash order within stratum").alias("selection_rule")),
                  dif.with_columns(pl.lit("difficult").alias("selection"),
                                   pl.lit("highest interior jerk RMS within stratum (A3)").alias("selection_rule"))]
    if not parts:
        return TASKS.empty()
    t = pl.concat(parts)
    split = assign_split(sorted(t["nfl_id"].unique().to_list()), cfg.dev_fraction, seed)
    t = t.with_columns(pl.col("nfl_id").replace_strict(split, return_dtype=pl.String).alias("split"))
    prio = {("eval", "random"): 1, ("eval", "difficult"): 2, ("dev", "random"): 3, ("dev", "difficult"): 4}
    t = t.with_columns(pl.struct("split", "selection").map_elements(lambda s: prio[(s["split"], s["selection"])],
                                                                    return_dtype=pl.Int64).alias("priority"))
    t = t.sort("priority", "stratum", "_hash")
    return t.select(pl.format("P{}", pl.int_range(1, pl.len() + 1).cast(pl.String).str.zfill(4)).alias("task_id"),
                    "domain", "traj_id", pl.col("nfl_id").cast(pl.Int64), "stratum", "position",
                    pl.col("cohort").cast(pl.Int64), "selection", "split", "priority", "selection_rule",
                    pl.col("duration_s").cast(pl.Float64))


# ============================================================================== 2. frames / detections
def _keys(domain: str) -> tuple[list[str], TrajectoryDomain]:
    return (CKEYS, COMBINE) if domain == "combine" else (NKEYS, NFL)


def task_raw(builder: KinematicFeatureBuilder, tasks: pl.DataFrame) -> dict[str, pl.DataFrame]:
    """Raw silver frames of the task trajectories only (bounded: ≤ a few hundred trajectories)."""
    out = {}
    c = tasks.filter(pl.col("domain") == "combine")
    if c.height:
        raw = builder.load_combine_raw().filter(pl.col("nfl_id").is_in(c["nfl_id"].unique().to_list()))
        out["combine"] = cast_keys(raw).with_columns(traj_id_expr(CKEYS)).filter(
            pl.col("traj_id").is_in(c["traj_id"].to_list())).drop("traj_id")
    n = tasks.filter(pl.col("domain") == "nfl_sample")
    if n.height:
        ids = n.select(pl.col("cohort").alias("season"), pl.col("traj_id").str.split("|").alias("p")).select(
            "season", pl.col("p").list.get(0).cast(pl.Int64).alias("game_id"),
            pl.col("p").list.get(1).cast(pl.Int64).alias("play_id"), pl.col("p").list.get(2).cast(pl.Int64).alias("nfl_id"))
        files = builder.nfl_season_files()
        parts = []
        for season, g in sorted(set(ids.select("season", "game_id").iter_rows())):   # NFL cohort = season
            if season not in files:
                continue
            r = builder.load_nfl_raw(files[season], season, g, g)
            if r.height:
                parts.append(r.join(ids.drop("season"), on=["game_id", "play_id", "nfl_id"], how="semi"))
        if parts:
            out["nfl_sample"] = pl.concat(parts, how="vertical_relaxed")
    return out


def task_kinematics(raw: dict[str, pl.DataFrame], kcfg: KinematicsConfig) -> dict[str, dict[str, pl.DataFrame]]:
    """{smoothing label: {'frames': …, 'events': …}} over all task trajectories (traj_id attached)."""
    out: dict[str, dict[str, pl.DataFrame]] = {}
    for spec in kcfg.sensitivity_smoothing:
        k = kcfg.with_smoothing(spec)
        fr, ev = [], []
        for dom, r in raw.items():
            keys, domain = _keys(dom)
            frames, _ = prepare_frames(r, domain, k)
            kin = compute_kinematics(frames, k)
            e = detect_events(kin, domain, k)
            fr.append(cast_keys(kin).with_columns(traj_id_expr(keys)).select(
                "traj_id", *[pl.col(c).cast(pl.Float64) for c in ("t_s", "speed", "accel_mag", "a_tan", "heading",
                                                                  "x_s", "y_s")], "valid_kin"))
            if e.height:
                ev.append(cast_keys(e).with_columns(traj_id_expr(keys)).select(
                    "traj_id", pl.col("rule").cast(pl.String), pl.col("event_type").cast(pl.String),
                    pl.col("t_start_s").cast(pl.Float64), pl.col("t_end_s").cast(pl.Float64)))
        out[spec.label] = {"frames": pl.concat(fr), "events": pl.concat(ev) if ev else pl.DataFrame(
            schema={"traj_id": S, "rule": S, "event_type": S, "t_start_s": F, "t_end_s": F})}
    return out


# ============================================================================== 3. evaluation (pure)
def evaluate_all(labels: pl.DataFrame, tasks: pl.DataFrame, detections: dict[str, pl.DataFrame],
                 cfg: AnnotationDesignConfig, seed: int) -> tuple[pl.DataFrame, pl.DataFrame]:
    """``detections``: smoothing label → (traj_id, rule, event_type, t_start_s, t_end_s)."""
    compat = tasks.select("task_id", "traj_id", "nfl_id", pl.col("stratum").alias("drill"), "position")
    meta = tasks.select("task_id", "split", "selection", "domain")
    matches, perf = [], []
    for sg, det in detections.items():
        for tol in cfg.tolerances_s:
            m = ann.match_all(labels, det, compat, tol)
            if not m.height:
                continue
            m = m.join(meta, on="task_id", how="left").with_columns(
                pl.lit(sg).alias("smoothing"), pl.col("rule").alias("detector_rule"), pl.lit(tol).alias("tolerance_s"))
            matches.append(m)
            acfg = ann.AnnotationConfig(tolerance_s=tol, n_boot=cfg.n_boot, seed=seed)
            for split in ("eval", "dev"):
                for sel in ("random", "difficult", "all"):
                    sub = m.filter((pl.col("split") == split) & ((pl.col("selection") == sel) | pl.lit(sel == "all")))
                    if not sub.height:
                        continue
                    p = ann.evaluate(sub.select(list(ann.MATCHES.columns)), acfg)
                    perf.append(p.rename({"rule": "detector_rule"}).with_columns(
                        pl.lit(split).alias("split"), pl.lit(sel).alias("selection_set"), pl.lit(sg).alias("smoothing")))
    mt = (pl.concat(matches, how="vertical").select(list(MATCHES.columns)) if matches else MATCHES.empty())
    pf = (pl.concat(perf, how="vertical_relaxed").select(list(PERFORMANCE.columns)) if perf else PERFORMANCE.empty())
    return mt, pf


def taxonomy(matches: pl.DataFrame, tol: float) -> pl.DataFrame:
    if not matches.height:
        return TAXONOMY.empty()
    return (matches.filter(pl.col("tolerance_s") == tol)
            .with_columns(pl.when(pl.col("error_type") == "").then(pl.lit("correct")).otherwise(pl.col("error_type"))
                          .alias("error_type"))
            .group_by("split", "smoothing", "detector_rule", "event_class", "outcome", "error_type")
            .agg(pl.len().cast(pl.Int64).alias("n")).sort("split", "smoothing", "detector_rule", "event_class"))


def coverage(labels: pl.DataFrame, tasks: pl.DataFrame) -> pl.DataFrame:
    done = set(labels["task_id"].unique().to_list()) if labels.height else set()
    return (tasks.with_columns(pl.col("task_id").is_in(list(done)).alias("_l"))
            .group_by("split", "selection", "stratum").agg(
                pl.len().cast(pl.Int64).alias("n_tasks"), pl.col("_l").sum().cast(pl.Int64).alias("n_labelled"))
            .with_columns((pl.col("n_labelled") / pl.col("n_tasks")).cast(pl.Float64).alias("share_labelled"))
            .sort("split", "selection", "stratum"))


# ============================================================================== orchestration
class AnnotationDesign:
    STUDY = STUDY

    def __init__(self, cfg: Phase01Config, pcfg: Phase04Config | None = None, store: StudyStore | None = None,
                 store03: StudyStore | None = None, kcfg: KinematicsConfig | None = None, **_: Any):
        self.cfg = cfg
        self.pcfg = pcfg or Phase04Config()
        self.a = self.pcfg.annotation
        self.kcfg = kcfg or KinematicsConfig.from_tracking(cfg.tracking)
        self.store = store or StudyStore(cfg, phase="phase_04")
        self.store03 = store03 or StudyStore(cfg)
        self.kin = KinematicFeatureBuilder(cfg, self.kcfg)
        self.root = cfg.paths.project_root / ANNOTATION_DIR

    def tasks(self) -> pl.DataFrame:
        """Write tasks, blind packets (primary SG) and the label template. Never touches a filled label file."""
        tasks = self.store.write(select_tasks(self.store03.read(SIGNALS), self.a, self.pcfg.seed), TASKS, STUDY)
        self.root.mkdir(parents=True, exist_ok=True)
        ann.label_template(tasks).write_csv(self.root / "annotation_labels_TEMPLATE.csv")
        primary_only = dataclasses.replace(self.kcfg, sensitivity_smoothing=(self.kcfg.smoothing,))
        kin = task_kinematics(task_raw(self.kin, tasks), primary_only)
        frames = kin[self.kcfg.smoothing.label]["frames"].filter(pl.col("valid_kin"))
        packets = ann.export_packets(frames, tasks.with_columns(pl.col("stratum").alias("drill")), self.root / "packets")
        print(f"{tasks.height} tasks ({tasks.group_by('split').len().rows()}), {len(packets)} packets → "
              f"{self.cfg.paths.rel(self.root)}. Copy annotation_labels_TEMPLATE.csv to annotation_labels.csv, "
              f"label in 'priority' order following docs/PHASE_04_ANNOTATION_PROTOCOL.md")
        return tasks

    def evaluate(self, labels_path: Path) -> dict[str, pl.DataFrame]:
        tasks = self.store.read(TASKS)
        raw = pl.read_csv(self.cfg.paths.project_root / labels_path,
                          schema_overrides={c: t for c, t in ann.LABELS.columns.items()})
        # template rows left blank (event_class empty) = task not labelled yet — not an error
        raw = raw.filter(pl.col("event_class").is_not_null() & (pl.col("event_class").str.strip_chars() != ""))
        labels = ann.validate_labels(raw, tasks.select("task_id", "duration_s"))
        labelled = tasks.filter(pl.col("task_id").is_in(labels["task_id"].unique().to_list()))
        kin = task_kinematics(task_raw(self.kin, labelled), self.kcfg)
        det = {sg: v["events"] for sg, v in kin.items()}
        m, perf = evaluate_all(labels, tasks, det, self.a, self.pcfg.seed)
        out = {MATCHES.name: self.store.write(m, MATCHES, STUDY),
               PERFORMANCE.name: self.store.write(perf, PERFORMANCE, STUDY),
               TAXONOMY.name: self.store.write(taxonomy(m, self.a.primary_tolerance_s), TAXONOMY, STUDY),
               COVERAGE.name: self.store.write(coverage(labels, tasks), COVERAGE, STUDY)}
        frames = kin[self.kcfg.smoothing.label]["frames"]
        for rec, fn in FIGURES:
            fn(self.store, rec, out, {"frames": frames, "labels": labels, "tasks": tasks, "detections": det,
                                      "kcfg": self.kcfg, "tol": self.a.primary_tolerance_s})
        headline = perf.filter((pl.col("split") == "eval") & (pl.col("selection_set") == "random")
                               & (pl.col("facet") == "all") & (pl.col("label_set") == "all_labels")
                               & (pl.col("tolerance_s") == self.a.primary_tolerance_s)
                               & (pl.col("smoothing") == self.kcfg.smoothing.label))
        with pl.Config(tbl_rows=30, tbl_cols=12):
            print(headline.select("detector_rule", "event_class", "n_tasks", "tp", "fp", "fn", "precision", "recall",
                                  "f1", "f1_lower", "f1_upper", "onset_mae_s"))
        return out


# ============================================================================== figures
def _style():
    from ...data.reporting import CATEGORICAL, INK

    return CATEGORICAL, INK


def fig_prf(store: StudyStore, rec: FigureRecord, out: dict, ctx: dict) -> None:
    from ..figkit import empty_panel, figure, finish

    cat, ink = _style()
    p = out[PERFORMANCE.name].filter((pl.col("split") == "eval") & (pl.col("selection_set") == "random")
                                     & (pl.col("facet") == "all") & (pl.col("label_set") == "all_labels")
                                     & (pl.col("tolerance_s") == ctx["tol"]))
    classes = [c for c in ("acceleration", "braking", "turn", "reversal") if c in set(p["event_class"].to_list())]
    fig, ax = figure(1, 3, sharey=True)
    for k, metric in enumerate(("precision", "recall", "f1")):
        a = ax[0, k]
        if not p.height or not classes:
            empty_panel(a, "no labelled eval-random tasks yet")
            continue
        configs = p.select("smoothing", "detector_rule").unique().sort("smoothing", "detector_rule").rows()
        for j, (sg, rule) in enumerate(configs):
            h = pl.DataFrame({"event_class": classes}).join(
                p.filter((pl.col("smoothing") == sg) & (pl.col("detector_rule") == rule)), on="event_class", how="left")
            y = np.arange(len(classes)) + (j - len(configs) / 2) * 0.07
            est = h[metric].fill_null(np.nan).to_numpy()
            lo, hi = h[f"{metric}_lower"].fill_null(np.nan).to_numpy(), h[f"{metric}_upper"].fill_null(np.nan).to_numpy()
            err = np.vstack([np.where(np.isfinite(lo), est - lo, 0), np.where(np.isfinite(hi), hi - est, 0)]).clip(0)
            a.errorbar(est, y, xerr=err, fmt="o" if sg == ctx["kcfg"].smoothing.label else ".", ms=4, capsize=1.5,
                       color=cat[["lenient", "default", "strict"].index(rule) if rule in ("lenient", "default", "strict")
                                 else 3], alpha=1.0 if sg == ctx["kcfg"].smoothing.label else 0.45,
                       label=f"{rule} · {sg}")
        a.set_yticks(np.arange(len(classes)), classes)
        a.invert_yaxis()
        a.set_xlim(0, 1.02)
        a.set_xlabel(metric)
    ax[0, -1].legend(frameon=False, fontsize=5.5, ncol=2)
    finish(fig, f"Eval split, random tasks (unbiased within strata), onset tolerance ± {ctx['tol']} s, all labels. Colour = "
                "rule; solid markers = primary SG setting, faint = alternative settings. Athlete-cluster bootstrap 95 % "
                "CI. PENDING interpretation.", "C2 · Detector precision, recall and F1 against human labels")
    store.save_figure(fig, rec, {"performance": p}, n=f"{int(p['n_tasks'].max() or 0) if p.height else 0} labelled tasks")


def fig_timing(store: StudyStore, rec: FigureRecord, out: dict, ctx: dict) -> None:
    from ..figkit import empty_panel, figure, finish

    cat, ink = _style()
    m = out[MATCHES.name].filter((pl.col("outcome") == "TP") & (pl.col("tolerance_s") == ctx["tol"])
                                 & (pl.col("smoothing") == ctx["kcfg"].smoothing.label) & (pl.col("split") == "eval"))
    fig, ax = figure(1, 2, sharey=True)
    for k, col in enumerate(("onset_error_s", "offset_error_s")):
        a = ax[0, k]
        if not m.height:
            empty_panel(a, "no matched events yet")
            continue
        for j, rule in enumerate(sorted(m["detector_rule"].unique().to_list())):
            v = m.filter(pl.col("detector_rule") == rule)[col].drop_nulls().to_numpy()
            if v.size:
                a.hist(v, bins=np.arange(-0.55, 0.56, 0.1), histtype="step", lw=1.5, color=cat[j],
                       label=f"{rule} (n={v.size})")
        a.axvline(0, color=ink["muted"], lw=0.8)
        a.set_xlabel(f"{col.replace('_', ' ')} (detector − reference)")
    ax[0, 0].set_ylabel("matched events")
    ax[0, -1].legend(frameon=False, fontsize=7)
    finish(fig, "Timing error of matched events (eval split, primary SG). Negative = detector earlier than the human "
                "onset. Bins of 0.1 s = one frame at 10 Hz. PENDING interpretation.", "C2 · Onset and offset error")
    store.save_figure(fig, rec, {"matches": m}, n=f"{m.height} matched events")


def fig_taxonomy(store: StudyStore, rec: FigureRecord, out: dict, ctx: dict) -> None:
    from ..figkit import empty_panel, figure, finish

    cat, ink = _style()
    t = out[TAXONOMY.name].filter((pl.col("split") == "eval") & (pl.col("smoothing") == ctx["kcfg"].smoothing.label)
                                  & (pl.col("outcome") != "TP"))
    fig, ax = figure(1, 2, sharey=True)
    for k, outcome in enumerate(("FP", "FN")):
        a = ax[0, k]
        g = t.filter(pl.col("outcome") == outcome)
        if not g.height:
            empty_panel(a, f"no {outcome} yet")
            continue
        rows = g.select("detector_rule", "event_class").unique().sort("event_class", "detector_rule").rows()
        types = sorted(g["error_type"].unique().to_list())
        left = np.zeros(len(rows))
        for j, et in enumerate(types):
            v = np.array([g.filter((pl.col("detector_rule") == r) & (pl.col("event_class") == c)
                                   & (pl.col("error_type") == et))["n"].sum() for r, c in rows], float)
            a.barh(np.arange(len(rows)), v, left=left, color=cat[j % len(cat)], label=et)
            left += v
        a.set_yticks(np.arange(len(rows)), [f"{c} · {r}" for r, c in rows], fontsize=7)
        a.invert_yaxis()
        a.set_xlabel(f"{outcome} count")
        a.legend(frameon=False, fontsize=7)
    finish(fig, "False positives: spurious, fragmentation (one human event split), class confusion. False negatives: "
                "missed, merged, timing (overlap but onset outside tolerance), class confusion. Eval split, primary SG. "
                "PENDING interpretation.", "C2 · What kind of errors does the detector make?")
    store.save_figure(fig, rec, {"taxonomy": out[TAXONOMY.name]}, n=f"{int(t['n'].sum()) if t.height else 0} errors")


def fig_tolerance(store: StudyStore, rec: FigureRecord, out: dict, ctx: dict) -> None:
    from ..figkit import empty_panel, figure, finish

    cat, ink = _style()
    p = out[PERFORMANCE.name].filter((pl.col("split") == "eval") & (pl.col("selection_set") == "random")
                                     & (pl.col("facet") == "all") & (pl.col("label_set") == "all_labels")
                                     & (pl.col("smoothing") == ctx["kcfg"].smoothing.label))
    classes = [c for c in ("acceleration", "braking", "turn", "reversal") if c in set(p["event_class"].to_list())]
    fig, ax = figure(1, max(1, len(classes)), sharey=True)
    for k, cls in enumerate(classes or ["none"]):
        a = ax[0, k]
        g = p.filter(pl.col("event_class") == cls)
        if not g.height:
            empty_panel(a, "no labels yet")
            continue
        for j, rule in enumerate(sorted(g["detector_rule"].unique().to_list())):
            h = g.filter(pl.col("detector_rule") == rule).sort("tolerance_s")
            a.plot(h["tolerance_s"], h["f1"], "o-", color=cat[j], label=rule)
        a.set_title(cls)
        a.set_xlabel("onset tolerance (s)")
        a.set_ylim(0, 1.02)
    ax[0, 0].set_ylabel("F1")
    ax[0, -1].legend(frameon=False, fontsize=7)
    finish(fig, "F1 as a function of the matching tolerance (eval split, random tasks, primary SG). A steep rise "
                "between 0.2 and 0.5 s means events are found but mistimed. PENDING interpretation.",
           "C2 · Sensitivity of detector accuracy to the matching tolerance")
    store.save_figure(fig, rec, {"performance": p}, n=f"{len(classes)} event classes")


def fig_examples(store: StudyStore, rec: FigureRecord, out: dict, ctx: dict) -> None:
    """Best (all TP) and worst (most FP + FN) labelled eval tasks under the primary configuration."""
    from ..figkit import empty_panel, figure, finish

    cat, ink = _style()
    sg, rule = ctx["kcfg"].smoothing.label, ctx["kcfg"].primary_rule
    m = out[MATCHES.name].filter((pl.col("smoothing") == sg) & (pl.col("detector_rule") == rule)
                                 & (pl.col("tolerance_s") == ctx["tol"]))
    fig, ax = figure(2, 1, size="tall")
    picks = []
    if m.height:
        sc = m.group_by("task_id").agg((pl.col("outcome") != "TP").sum().alias("err"), pl.len().alias("n"))
        best = sc.filter(pl.col("err") == 0).sort("n", "task_id", descending=[True, False]).head(1)
        worst = sc.sort("err", "task_id", descending=[True, False]).head(1)
        picks = [("correct", best), ("incorrect", worst)]
    rows = []
    for k in range(2):
        a = ax[k, 0]
        if k >= len(picks) or not picks[k][1].height:
            empty_panel(a, "no labelled task to show yet")
            continue
        label, sel = picks[k]
        tid = sel["task_id"][0]
        traj = ctx["tasks"].filter(pl.col("task_id") == tid)["traj_id"][0]
        f = ctx["frames"].filter(pl.col("traj_id") == traj).sort("t_s")
        a.plot(f["t_s"], f["speed"], color=ink["primary"], lw=1.2)
        a.set_ylabel("speed (yd/s)")
        ref = ctx["labels"].filter((pl.col("task_id") == tid) & (pl.col("event_class") != "none"))
        det = ctx["detections"][sg].filter((pl.col("traj_id") == traj) & (pl.col("rule") == rule))
        ymax = float(f["speed"].max() or 1.0)
        for r in ref.iter_rows(named=True):
            a.axvspan(r["onset_s"], r["offset_s"], ymin=0.5, ymax=1.0, color=cat[0], alpha=0.2)
            a.text(r["onset_s"], ymax * 0.98, r["event_class"][:4], fontsize=6, color=cat[0], va="top")
        for r in det.iter_rows(named=True):
            a.axvspan(r["t_start_s"], r["t_end_s"], ymin=0.0, ymax=0.5, color=cat[1], alpha=0.2)
            a.text(r["t_start_s"], 0.02 * ymax, r["event_type"][:4], fontsize=6, color=cat[1])
        a.set_title(f"{label}: {tid} · {traj} — top band = human, bottom band = detector ({rule}, {sg})", fontsize=9)
        rows.append({"example": label, "task_id": tid, "traj_id": traj})
    ax[1, 0].set_xlabel("t (s since trajectory start)")
    finish(fig, "Representative correct and incorrect detections among labelled tasks (deterministic: all-TP task with "
                "most events; task with most FP + FN). PENDING interpretation.", "C2 · Example detections vs labels")
    store.save_figure(fig, rec, {"examples": pl.DataFrame(rows) if rows else pl.DataFrame({"example": []})},
                      n=f"{len(rows)} tasks")


def _rec(fid: str, name: str, q: str, insight: str, stat: str, ds: str, grain: str) -> FigureRecord:
    return FigureRecord(figure_id=fid, name=name, study=STUDY, research_question=q, expected_insight=insight,
                        statistic=stat, dataset=ds, grain=grain, filters="labelled tasks only",
                        method="one-to-one onset matching vs human labels")


FIGURES = [
    (_rec("C2-01", "C2_01_precision_recall_f1", "How accurate is each detector configuration vs human labels?",
          "validated vs unvalidated event types", "P / R / F1 + cluster CI", PERFORMANCE.name, "event"), fig_prf),
    (_rec("C2-02", "C2_02_timing_error", "How well timed are detected events?", "onset / offset bias",
          "error histogram", MATCHES.name, "event"), fig_timing),
    (_rec("C2-03", "C2_03_error_taxonomy", "Which error types dominate?", "detector failure modes",
          "counts", TAXONOMY.name, "event"), fig_taxonomy),
    (_rec("C2-04", "C2_04_tolerance_sensitivity", "Does accuracy depend on the matching tolerance?",
          "found-but-mistimed events", "F1 vs tolerance", PERFORMANCE.name, "event"), fig_tolerance),
    (_rec("C2-05", "C2_05_example_detections", "What do correct and incorrect detections look like?",
          "visual audit", "time series + spans", MATCHES.name, "frame"), fig_examples),
]
