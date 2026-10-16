# Phase 04 · Study C2 v2 — event annotation protocol

_Applies to `reports/annotation/phase_04/annotation_labels.csv`. Written before any labelling. Definitions are fixed now so that the detector, which you have not seen, cannot shape them._

## 1. Material
For each task there are two files:
- `packets/<task_id>.png`: speed, tangential acceleration and |a| against time, plus the path. It is **blind**: no detector output is shown.
- `packets/<task_id>.csv`: the same signals, so exact times can be read.

The time base is `t_s` = seconds since the trajectory's first frame. Record onset and offset in this time base, to 0.1 s (one frame).

## 2. Order
Label in `priority` order:
1. eval / random
2. eval / difficult
3. dev / random
4. dev / difficult

The headline accuracy is the **eval / random** set. Do not open any detector output before the eval set is finished. The dev split exists so a detector revision can be tuned without touching eval athletes.

## 3. Event classes and boundaries
| Class | Onset | Offset | Not this class |
|---|---|---|---|
| `acceleration` | first frame where speed starts a sustained rise (≥ 0.3 s, gain ≥ ~1 yd/s) | frame where the rise ends (plateau or peak) | the slow drift in the forty's flying start |
| `braking` | first frame of a sustained speed drop (≥ 0.3 s, loss ≥ ~1 yd/s) from ≥ 2 yd/s | minimum speed, or the end of the drop | the deceleration after the finish line, if it is in the recording |
| `turn` | heading begins to change at running speed (≥ 2 yd/s) | heading change completes | a change of direction through near-zero speed (that is a `reversal`) |
| `reversal` | speed falls below ~1 yd/s before a plant-and-cut | speed rises again above ~1 yd/s in the new direction (≥ 90° change) | a full stop without a direction change |
| `reacceleration` | first frame of the speed rise **after** a braking / reversal | end of that rise | the initial start from rest (that is `acceleration`) |
| `none` | one row per task with no events | — | — |

**Rules:**
- An event may overlap events of other classes. For example, braking → reversal → reacceleration is three rows.
- Do not label an event that is cut off at the start or end of the recording unless both its onset and its offset are visible. Mention a cut-off event in `notes`.
- Use `confidence = uncertain` when the boundary or the class is ambiguous. Evaluation reports results with and without uncertain labels.
- **NFL plays:** label the entire tracked window, including pre-snap motion. The analysis separates windows later, so the annotation stays window-agnostic.

## 4. File format
Copy `annotation_labels_TEMPLATE.csv` to `annotation_labels.csv`. The template has one row per task. Add rows for further events, and number them with `label_idx` = 1, 2, … within a task.

| Column | Value |
|---|---|
| task_id | from the template |
| annotator | your initials (the same for every row you label) |
| label_idx | 1, 2, … |
| event_class | one of the classes above |
| onset_s / offset_s | seconds, `t_s` time base (empty for `none`) |
| confidence | `certain` / `uncertain` |
| notes | free text |

Rows of tasks you have not labelled yet can stay blank, because blank `event_class` rows are ignored. Validation rejects:
- unknown classes;
- onset ≥ offset;
- events outside the trajectory;
- `none` mixed with real events;
- duplicate indices.

## 5. Matching and metrics
- A detection matches a reference event of the same class when |Δonset| ≤ 0.3 s. Results are also reported at 0.2 s and 0.5 s.
- Matching is one-to-one and greedy, with deterministic tie-breaking.
- `reacceleration` is scored as detector class `acceleration`.
- Metrics:
  - precision, recall and F1, with athlete-cluster bootstrap CIs;
  - onset and offset error, and count error;
  - an FP / FN taxonomy: spurious, fragmentation, merged, timing, class confusion, missed.
- Every metric is reported per SG setting × rule.

## 6. Second annotator (recommended)
If a second person labels a subset (≥ 20 eval tasks), inter-annotator agreement becomes the ceiling for detector accuracy. The current evaluation uses the alphabetically first annotator per task; an agreement table is a planned extension.
