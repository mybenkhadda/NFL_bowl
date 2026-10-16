"""Study C2 — reference annotation workflow and detector evaluation.

Workflow (all steps are code; the human labelling is done by the user)::

    1. select_tasks()            deterministic, stratified task list (drill × cohort × position;
                                 'typical' and 'difficult' trajectories) → annotation_tasks.csv
    2. export_packets()          per task: signal CSV + a BLIND figure (speed, |a|, heading, path — NO
                                 detector output, to avoid anchoring) → reports/annotation/packets/
    3. user fills                annotation_labels.csv (schema ``LABELS``; template written in step 1)
    4. validate_labels()         schema + logical checks (onset < offset, inside the trajectory,
                                 explicit 'none' rows for trajectories without events)
    5. match_events()            one-to-one greedy matching of detector events to reference events
                                 of the same class by |Δonset| ≤ tolerance
    6. evaluate()                precision / recall / F1, onset & offset error, count error, error
                                 taxonomy; athlete-cluster bootstrap CIs; with and without
                                 'uncertain' labels; per drill / position / rule

Classes: acceleration, braking, turn, reversal, reacceleration (annotators). The M1 detector has no
'reacceleration' class — for detector evaluation it is mapped to 'acceleration' (``DETECTOR_CLASS``),
and C3 (maneuvers) evaluates it separately.

Time base: ``t_s`` = seconds since the trajectory's first frame (the M1 convention used by
``combine_kinematic_events.t_start_s / t_end_s``), so labels and detections are directly comparable.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

from ..contracts import ContractError, TableContract
from .stats import resample_counts

S, I, F, B = pl.String(), pl.Int64(), pl.Float64(), pl.Boolean()
CLASSES = ("acceleration", "braking", "turn", "reversal", "reacceleration", "none")
DETECTOR_CLASS = {"acceleration": "acceleration", "braking": "braking", "turn": "turn", "reversal": "reversal",
                  "reacceleration": "acceleration"}
CONFIDENCE = ("certain", "uncertain")

TASKS = TableContract(
    "annotation_tasks", "Deterministic, stratified annotation task list (blind to detector output).",
    {"task_id": S, "domain": S, "traj_id": S, "nfl_id": I, "drill": S, "cohort": I, "position": S,
     "selection": S, "selection_rule": S, "duration_s": F}, key=("task_id",), area="hypothesis_screening")
LABELS = TableContract(
    "annotation_labels", "Human reference events (one row per event; event_class='none' = no events).",
    {"task_id": S, "annotator": S, "label_idx": I, "event_class": S, "onset_s": F, "offset_s": F, "confidence": S,
     "notes": S}, key=("task_id", "annotator", "label_idx"), area="hypothesis_screening")
MATCHES = TableContract(
    "annotation_matches", "Reference ↔ detector matches, misses and false alarms with error taxonomy.",
    {"task_id": S, "nfl_id": I, "drill": S, "position": S, "rule": S, "event_class": S, "outcome": S,
     "error_type": S, "ref_onset_s": F, "ref_offset_s": F, "det_onset_s": F, "det_offset_s": F, "onset_error_s": F,
     "offset_error_s": F, "ref_uncertain": B}, area="hypothesis_screening")
PERFORMANCE = TableContract(
    "detector_performance", "Detector precision / recall / F1 / timing error with athlete-cluster bootstrap CIs.",
    {"rule": S, "event_class": S, "facet": S, "level": S, "label_set": S, "n_tasks": I, "n_athletes": I, "tp": I,
     "fp": I, "fn": I, "precision": F, "precision_lower": F, "precision_upper": F, "recall": F, "recall_lower": F,
     "recall_upper": F, "f1": F, "f1_lower": F, "f1_upper": F, "onset_mae_s": F, "onset_median_s": F,
     "offset_mae_s": F, "count_mae": F, "tolerance_s": F},
    key=("rule", "event_class", "facet", "level", "label_set"), area="hypothesis_screening")


@dataclass(frozen=True)
class AnnotationConfig:
    drills: tuple[str, ...] = ("FORTY_YARD_DASH", "SHORT_SHUTTLE", "THREE_CONE_DRILL")
    other_drills_per_cohort: int = 4          # extra tasks from skill drills (diversity of movement content)
    per_stratum_typical: int = 2              # per drill × cohort
    per_stratum_difficult: int = 2
    tolerance_s: float = 0.3                  # PROPOSED onset tolerance (3 frames at 10 Hz)
    n_boot: int = 1000
    seed: int = 20261008


# ----------------------------------------------------------------------------- 1. task selection
def select_tasks(dims: pl.DataFrame, cfg: AnnotationConfig = AnnotationConfig()) -> pl.DataFrame:
    """``dims``: Study A ``trajectory_quality_dimensions`` (Combine rows) joined with ``drill``.

    typical   = event_count_rel_range closest to the stratum median (detector rule choice matters little)
    difficult = largest event_count_rel_range (rule choice changes the event count most)
    Ties → traj_id. Tier 'reject' is excluded (no valid kinematics). Outcome-blind: no official results used.
    """
    d = dims.filter((pl.col("domain") == "combine") & (pl.col("quality_tier") != "reject"))
    if "drill" not in d.columns:   # Study A stores the Combine drill type as 'stratum'
        d = d.with_columns(pl.col("stratum").alias("drill"))
    key = pl.col("event_count_rel_range").fill_null(0.0)
    main = d.filter(pl.col("drill").is_in(list(cfg.drills)))
    parts = []
    for (drill, cohort), g in main.group_by("drill", "cohort", maintain_order=True):
        med = g.select(key.median()).item()
        typ = g.sort([(key - med).abs(), pl.col("traj_id")]).head(cfg.per_stratum_typical)
        dif = (g.filter(~pl.col("traj_id").is_in(typ["traj_id"].to_list()))
               .sort([key, pl.col("traj_id")], descending=[True, False]).head(cfg.per_stratum_difficult))
        parts += [typ.with_columns(pl.lit("typical").alias("selection"),
                                   pl.lit("event-count rule sensitivity closest to drill × cohort median").alias("selection_rule")),
                  dif.with_columns(pl.lit("difficult").alias("selection"),
                                   pl.lit("largest event-count rule sensitivity in drill × cohort").alias("selection_rule"))]
    other = d.filter(~pl.col("drill").is_in(list(cfg.drills)))
    for (cohort,), g in other.group_by("cohort", maintain_order=True):
        parts.append(g.sort([key, pl.col("traj_id")], descending=[True, False]).head(cfg.other_drills_per_cohort)
                     .with_columns(pl.lit("difficult").alias("selection"),
                                   pl.lit("largest event-count rule sensitivity among skill drills").alias("selection_rule")))
    if not parts:
        return TASKS.empty()
    t = pl.concat(parts).unique("traj_id", keep="first", maintain_order=True).sort("drill", "cohort", "selection", "traj_id")
    return t.select(pl.format("T{}", pl.int_range(1, pl.len() + 1).cast(pl.String).str.zfill(4)).alias("task_id"),
                    "domain", "traj_id", "nfl_id", "drill", "cohort", "position", "selection", "selection_rule",
                    "duration_s")


def label_template(tasks: pl.DataFrame) -> pl.DataFrame:
    """Empty-row template for the annotator (one placeholder row per task)."""
    return pl.DataFrame({"task_id": tasks["task_id"], "annotator": [""] * tasks.height,
                         "label_idx": [1] * tasks.height, "event_class": [""] * tasks.height,
                         "onset_s": [None] * tasks.height, "offset_s": [None] * tasks.height,
                         "confidence": ["certain"] * tasks.height, "notes": [""] * tasks.height},
                        schema=LABELS.columns)


# ----------------------------------------------------------------------------- 2. packets
def export_packets(frames: pl.DataFrame, tasks: pl.DataFrame, out_dir: Path) -> list[Path]:
    """Per task: ``<task_id>.csv`` (t_s, speed, accel_mag, a_tan, heading, x_s, y_s) and a BLIND figure."""
    from ..figkit import figure, finish

    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for r in tasks.iter_rows(named=True):
        f = frames.filter(pl.col("traj_id") == r["traj_id"]).sort("t_s")
        if not f.height:
            continue
        sig = f.select("t_s", "speed", "accel_mag", "a_tan", "heading", "x_s", "y_s")
        p = out_dir / f"{r['task_id']}.csv"
        sig.write_csv(p)
        fig, ax = figure(3, 1, size="tall", sharex=False)
        ax[0, 0].plot(sig["t_s"], sig["speed"])
        ax[0, 0].set_ylabel("speed (yd/s)")
        ax[1, 0].plot(sig["t_s"], sig["a_tan"], label="tangential a")
        ax[1, 0].plot(sig["t_s"], sig["accel_mag"], alpha=0.6, label="|a|")
        ax[1, 0].set_ylabel("yd/s²")
        ax[1, 0].legend(fontsize=8, frameon=False)
        ax[2, 0].plot(sig["x_s"], sig["y_s"])
        ax[2, 0].set_aspect("equal", adjustable="datalim")
        ax[2, 0].set_xlabel("x (yd)")
        for a in ax[:2, 0]:
            a.set_xticks(np.arange(0, float(sig["t_s"].max() or 0) + 0.5, 0.5), minor=True)
            a.grid(which="minor", alpha=0.2)
        ax[1, 0].set_xlabel("t_s (s since trajectory start) — record onset/offset in this time base")
        finish(fig, f"{r['task_id']} · {r['drill']} · blind packet (no detector output)", title=f"Annotation {r['task_id']}")
        fig.savefig(out_dir / f"{r['task_id']}.png", dpi=150)
        written.append(p)
    return written


# ----------------------------------------------------------------------------- 4. validation
def validate_labels(labels: pl.DataFrame, tasks: pl.DataFrame) -> pl.DataFrame:
    """Raise ``ContractError`` with every problem found; returns the validated labels."""
    problems = []
    missing = [c for c in LABELS.columns if c not in labels.columns]
    if missing:
        raise ContractError(f"annotation_labels: missing columns {missing}")
    lab = labels.select([pl.col(c).cast(t, strict=True) for c, t in LABELS.columns.items()])   # bad values raise
    bad_cls = lab.filter(~pl.col("event_class").is_in(list(CLASSES)))
    if bad_cls.height:
        problems.append(f"{bad_cls.height} rows with unknown event_class (allowed {CLASSES})")
    bad_conf = lab.filter(~pl.col("confidence").is_in(list(CONFIDENCE)))
    if bad_conf.height:
        problems.append(f"{bad_conf.height} rows with confidence not in {CONFIDENCE}")
    ev = lab.filter(pl.col("event_class") != "none")
    if ev.filter(pl.col("onset_s").is_null() | pl.col("offset_s").is_null()).height:
        problems.append("events without onset/offset")
    if ev.filter(pl.col("onset_s") >= pl.col("offset_s")).height:
        problems.append("events with onset ≥ offset")
    dur = ev.join(tasks.select("task_id", "duration_s"), on="task_id", how="left")
    if dur.filter(pl.col("duration_s").is_null()).height:
        problems.append("labels for unknown task_id")
    if dur.filter((pl.col("onset_s") < -0.05) | (pl.col("offset_s") > pl.col("duration_s") + 0.15)).height:
        problems.append("events outside the trajectory time range")
    none_mixed = lab.group_by("task_id", "annotator").agg(
        ((pl.col("event_class") == "none").any() & (pl.col("event_class") != "none").any()).alias("mixed"))
    if none_mixed.filter(pl.col("mixed")).height:
        problems.append("tasks mixing 'none' with real events")
    if lab.select(pl.struct("task_id", "annotator", "label_idx").is_duplicated().any()).item():
        problems.append("duplicate (task_id, annotator, label_idx)")
    if problems:
        raise ContractError("annotation_labels: " + "; ".join(problems))
    return lab   # partially completed annotation is allowed; evaluation uses labelled tasks only


# ----------------------------------------------------------------------------- 5. matching
def match_events(ref: list[tuple[float, float, bool]], det: list[tuple[float, float]], tol: float
                 ) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """One-to-one greedy matching by smallest |Δonset| ≤ ``tol``. ``ref`` items: (onset, offset, uncertain).
    Returns (matched (ref_i, det_j), unmatched ref indices, unmatched det indices); deterministic ties
    (by Δ, then ref index, then det index)."""
    cand = sorted((abs(r[0] - d[0]), i, j) for i, r in enumerate(ref) for j, d in enumerate(det)
                  if abs(r[0] - d[0]) <= tol)
    used_r, used_d, pairs = set(), set(), []
    for _, i, j in cand:
        if i not in used_r and j not in used_d:
            used_r.add(i)
            used_d.add(j)
            pairs.append((i, j))
    return (sorted(pairs), [i for i in range(len(ref)) if i not in used_r],
            [j for j in range(len(det)) if j not in used_d])


def _overlaps(a: tuple[float, float], b: tuple[float, float]) -> bool:
    return a[0] < b[1] and b[0] < a[1]


def classify_errors(ref_all: dict[str, list], det_all: dict[str, list], tol: float) -> list[dict]:
    """Per class: TP / FN / FP rows with taxonomy.

    FP 'fragmentation'   unmatched detection overlapping a reference of the same class that IS matched
    FP 'class_confusion' unmatched detection overlapping a reference of another class
    FP 'spurious'        otherwise
    FN 'merged'          unmatched reference overlapping a detection matched to another reference
    FN 'class_confusion' unmatched reference overlapping a detection of another class
    FN 'timing'          a same-class detection overlaps but its onset is outside the tolerance
    FN 'missed'          otherwise
    """
    rows = []
    for cls in sorted(set(ref_all) | set(det_all)):
        ref, det = ref_all.get(cls, []), det_all.get(cls, [])
        pairs, fn, fp = match_events(ref, det, tol)
        matched_det = {j for _, j in pairs}
        others_det = [d for c, v in det_all.items() if c != cls for d in v]
        others_ref = [r for c, v in ref_all.items() if c != cls for r in v]
        for i, j in pairs:
            rows.append({"event_class": cls, "outcome": "TP", "error_type": "", "ref": ref[i], "det": det[j]})
        for i in fn:
            r = ref[i]
            if any(_overlaps(r[:2], det[j]) for j in matched_det):
                et = "merged"
            elif any(_overlaps(r[:2], d) for d in det):
                et = "timing"
            elif any(_overlaps(r[:2], d) for d in others_det):
                et = "class_confusion"
            else:
                et = "missed"
            rows.append({"event_class": cls, "outcome": "FN", "error_type": et, "ref": r, "det": None})
        matched_ref = [ref[i] for i, _ in pairs]
        for j in fp:
            d = det[j]
            if any(_overlaps(d, r[:2]) for r in matched_ref):
                et = "fragmentation"
            elif any(_overlaps(d, r[:2]) for r in others_ref):
                et = "class_confusion"
            else:
                et = "spurious"
            rows.append({"event_class": cls, "outcome": "FP", "error_type": et, "ref": None, "det": d})
    return rows


def match_all(labels: pl.DataFrame, detections: pl.DataFrame, tasks: pl.DataFrame, tol: float,
              annotator: str | None = None) -> pl.DataFrame:
    """``detections``: traj_id, rule, event_type, t_start_s, t_end_s (M1 events, any rules).
    Only labelled tasks are evaluated; one annotator at a time (default: first alphabetically per task)."""
    lab = labels.filter(pl.col("annotator") == annotator) if annotator else (
        labels.filter(pl.col("annotator") == pl.col("annotator").min().over("task_id")))
    lab = lab.with_columns(pl.col("event_class").replace_strict(DETECTOR_CLASS, default="none").alias("det_class"))
    labelled = tasks.filter(pl.col("task_id").is_in(lab["task_id"].unique().to_list()))
    out = []
    for rule in sorted(detections["rule"].unique().to_list()):
        dr = detections.filter(pl.col("rule") == rule)
        for t in labelled.iter_rows(named=True):
            ref_all: dict[str, list] = {}
            for r in lab.filter((pl.col("task_id") == t["task_id"]) & (pl.col("det_class") != "none")).iter_rows(named=True):
                ref_all.setdefault(r["det_class"], []).append((r["onset_s"], r["offset_s"], r["confidence"] == "uncertain"))
            det_all: dict[str, list] = {}
            for d in dr.filter(pl.col("traj_id") == t["traj_id"]).sort("t_start_s").iter_rows(named=True):
                det_all.setdefault(d["event_type"], []).append((d["t_start_s"], d["t_end_s"]))
            for e in classify_errors(ref_all, det_all, tol):
                r, d = e["ref"], e["det"]
                out.append({"task_id": t["task_id"], "nfl_id": t["nfl_id"], "drill": t["drill"],
                            "position": t["position"], "rule": rule, "event_class": e["event_class"],
                            "outcome": e["outcome"], "error_type": e["error_type"],
                            "ref_onset_s": r[0] if r else None, "ref_offset_s": r[1] if r else None,
                            "det_onset_s": d[0] if d else None, "det_offset_s": d[1] if d else None,
                            "onset_error_s": (d[0] - r[0]) if r and d else None,
                            "offset_error_s": (d[1] - r[1]) if r and d else None,
                            "ref_uncertain": bool(r[2]) if r else False})
    return pl.DataFrame(out, schema=MATCHES.columns) if out else MATCHES.empty()


# ----------------------------------------------------------------------------- 6. evaluation
def _prf_boot(per: pl.DataFrame, n_boot: int, seed: int) -> dict[str, float]:
    tp, fp, fn = (per[c].to_numpy().astype(float) for c in ("tp", "fp", "fn"))
    out = {}
    with np.errstate(invalid="ignore", divide="ignore"):
        est = {"precision": tp.sum() / (tp.sum() + fp.sum()), "recall": tp.sum() / (tp.sum() + fn.sum()),
               "f1": 2 * tp.sum() / (2 * tp.sum() + fp.sum() + fn.sum())}
        if tp.size >= 2:
            w = resample_counts(tp.size, n_boot, seed)
            T, P, N = w @ tp, w @ fp, w @ fn
            reps = {"precision": T / (T + P), "recall": T / (T + N), "f1": 2 * T / (2 * T + P + N)}
        else:
            reps = {k: np.array([np.nan]) for k in est}
    for k, v in est.items():
        r = reps[k][np.isfinite(reps[k])]
        out[k] = float(v) if np.isfinite(v) else None
        out[f"{k}_lower"], out[f"{k}_upper"] = (tuple(float(x) for x in np.quantile(r, [0.025, 0.975]))
                                                if r.size > 1 else (None, None))
    return out


def evaluate(matches: pl.DataFrame, cfg: AnnotationConfig = AnnotationConfig()) -> pl.DataFrame:
    rows = []
    sets = {"all_labels": matches,
            "certain_only": matches.filter(~(pl.col("ref_uncertain") & (pl.col("outcome") != "FP")))}
    for label_set, m0 in sets.items():
        m0 = m0.with_columns(pl.lit("all").alias("all"))
        for facet in ("all", "drill", "position"):
            for (rule, cls, level), m in m0.group_by("rule", "event_class", facet, maintain_order=True):
                per = (m.group_by("nfl_id").agg(*[(pl.col("outcome") == o).sum().alias(o.lower()) for o in ("TP", "FP", "FN")])
                       .sort("nfl_id"))
                cnt = (m.group_by("task_id").agg(((pl.col("outcome") != "FN").cast(pl.Int64).sum()
                                                   - (pl.col("outcome") != "FP").cast(pl.Int64).sum()).abs().alias("ce")))
                tp = m.filter(pl.col("outcome") == "TP")
                rows.append({"rule": rule, "event_class": cls, "facet": facet, "level": str(level), "label_set": label_set,
                             "n_tasks": m["task_id"].n_unique(), "n_athletes": per.height, "tp": int(per["tp"].sum()),
                             "fp": int(per["fp"].sum()), "fn": int(per["fn"].sum()),
                             **_prf_boot(per, cfg.n_boot, cfg.seed),
                             "onset_mae_s": tp["onset_error_s"].abs().mean(), "onset_median_s": tp["onset_error_s"].median(),
                             "offset_mae_s": tp["offset_error_s"].abs().mean(), "count_mae": cnt["ce"].mean(),
                             "tolerance_s": cfg.tolerance_s})
    return pl.DataFrame(rows, schema=PERFORMANCE.columns, strict=False) if rows else PERFORMANCE.empty()
