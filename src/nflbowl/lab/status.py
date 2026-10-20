"""Explicit research status vocabulary and a guarded step runner shared by Milestones 15–18.

Every experiment step ends in exactly one status; nothing is silently skipped:

    READY                         inputs present, not yet run in this run directory
    COMPLETED                     ran in this session and wrote its outputs
    CACHED                        outputs reloaded from this run directory (provenance / signature matched)
    BLOCKED_MISSING_INPUT         a required file, column or optional package is absent (the reason names it)
    BLOCKED_PROVENANCE            an artifact exists but cannot be shown to be fold-safe / compatible
    BLOCKED_INSUFFICIENT_SAMPLE   too few athletes / trials / linked athletes for a meaningful estimate
    FAILED_VALIDATION             a scientific or data-contract check failed (the run continues; the result is unusable)
    FAILED                        an unexpected exception (traceback in ``logs/``)
    NOT_RUN                       disabled by configuration, or a prerequisite step did not complete

Scientific gates use their own default ``NOT_EVALUATED``: a gate becomes PASS / FAIL only from computed tables, never
from code, configuration or cached metadata alone (``gate_row``).

``run_step`` wraps ``LabRun.run_stage`` (stage ledger ``manifests/stages.json``, tracebacks in ``logs/``) and additionally
records the status above in ``manifests/<ledger>_status.json`` so a later Colab session sees exactly what is missing.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

from ..artifacts.store import atomic_json
from ..utils.reproducibility import utc_now
from .store import StageSkipped

READY = "READY"
COMPLETED = "COMPLETED"
CACHED = "CACHED"
BLOCKED_MISSING_INPUT = "BLOCKED_MISSING_INPUT"
BLOCKED_PROVENANCE = "BLOCKED_PROVENANCE"
BLOCKED_INSUFFICIENT_SAMPLE = "BLOCKED_INSUFFICIENT_SAMPLE"
FAILED_VALIDATION = "FAILED_VALIDATION"
FAILED = "FAILED"
NOT_RUN = "NOT_RUN"
STEP_STATUSES = (READY, COMPLETED, CACHED, BLOCKED_MISSING_INPUT, BLOCKED_PROVENANCE, BLOCKED_INSUFFICIENT_SAMPLE, FAILED_VALIDATION, FAILED,
                 NOT_RUN)
BLOCKED = (BLOCKED_MISSING_INPUT, BLOCKED_PROVENANCE, BLOCKED_INSUFFICIENT_SAMPLE)
DONE = (COMPLETED, CACHED)

# gate statuses (scientific decisions)
NOT_EVALUATED = "NOT_EVALUATED"
PASS = "PASS"
FAIL = "FAIL"
GATE_STATUSES = (NOT_EVALUATED, PASS, FAIL, *BLOCKED)


class ResearchBlocked(StageSkipped):
    """A step cannot run for a documented reason. ``status`` is one of ``BLOCKED``; the message names the missing input."""

    def __init__(self, status: str, reason: str, missing: Sequence[str] = (), detail: dict[str, Any] | None = None):
        if status not in BLOCKED:
            raise ValueError(f"ResearchBlocked status must be one of {BLOCKED}, got {status!r}")
        self.status, self.reason, self.missing, self.detail = status, reason, tuple(missing), dict(detail or {})
        extra = f" (missing: {', '.join(self.missing)})" if self.missing else ""
        super().__init__(f"blocked: {status}: {reason}{extra}")


class ValidationFailed(AssertionError):
    """A data-contract or scientific check failed. ``detail`` holds the evidence (written to the status ledger)."""

    def __init__(self, reason: str, detail: dict[str, Any] | None = None):
        self.reason, self.detail = reason, dict(detail or {})
        super().__init__(reason)


@dataclass
class StatusLedger:
    """``manifests/<name>_status.json``: one record per step (status, reason, outputs, timing, detail)."""

    path: Path
    records: dict[str, dict[str, Any]] = field(default_factory=dict)

    @classmethod
    def open(cls, run_root: Path | str, name: str) -> "StatusLedger":
        p = Path(run_root) / "manifests" / f"{name}_status.json"
        recs: dict[str, dict[str, Any]] = {}
        if p.is_file():
            try:
                recs = json.loads(p.read_text(encoding="utf-8")).get("steps", {})
            except (OSError, json.JSONDecodeError):
                recs = {}
        return cls(p, recs)

    def set(self, step: str, status: str, reason: str = "", outputs: Sequence[str] = (), **extra: Any) -> dict[str, Any]:
        if status not in STEP_STATUSES:
            raise ValueError(f"unknown step status {status!r}; one of {STEP_STATUSES}")
        rec = {"step": step, "status": status, "reason": reason, "outputs": list(outputs), "updated_at": utc_now(), **extra}
        self.records[step] = rec
        self.save()
        return rec

    def get(self, step: str) -> str:
        return self.records.get(step, {}).get("status", NOT_RUN)

    def done(self, step: str) -> bool:
        return self.get(step) in DONE

    def save(self) -> Path:
        return atomic_json({"updated_at": utc_now(), "steps": self.records}, self.path, overwrite=True)

    def table(self) -> Any:
        import polars as pl

        rows = [{"step": k, "status": v.get("status"), "reason": v.get("reason", ""), "n_outputs": len(v.get("outputs", [])),
                 "seconds": v.get("seconds"), "updated_at": v.get("updated_at")} for k, v in sorted(self.records.items())]
        return pl.DataFrame(rows, infer_schema_length=None) if rows else pl.DataFrame(
            schema={"step": pl.String, "status": pl.String, "reason": pl.String, "n_outputs": pl.Int64, "seconds": pl.Float64, "updated_at": pl.String})


def run_step(run: Any, ledger: StatusLedger, step: str, fn: Callable[[dict[str, Any]], Any], note: str = "", inputs: Sequence[str] = (),
             enabled: bool = True, reason_disabled: str = "", requires: Sequence[str] = ()) -> Any:
    """Run ``fn(ctx)`` as a guarded stage and record its status.

    * ``requires``: steps that must be COMPLETED / CACHED first; otherwise the step is NOT_RUN with the unmet list.
    * ``fn`` may set ``ctx['cache'] = 'hit'`` (→ CACHED) and append written paths to ``ctx['outputs']``.
    * ``ResearchBlocked`` → its BLOCKED_* status; ``ValidationFailed`` → FAILED_VALIDATION; any other exception → FAILED.
      Failures are recorded, the traceback is written by ``LabRun.run_stage``, and later independent steps still run.
    Returns ``fn``'s result, or None when the step did not complete.
    """
    unmet = [r for r in requires if not ledger.done(r)]
    if not enabled or unmet:
        if not enabled:
            why = reason_disabled or "disabled by configuration"
        else:
            why = f"prerequisite step(s) not completed: {', '.join(unmet)}"
        ledger.set(step, NOT_RUN, why)
        print(f"[{step}] NOT_RUN: {why}")
        return None
    box: dict[str, Any] = {}

    def wrapped(ctx: dict[str, Any]) -> Any:
        try:
            out = fn(ctx)
        except ResearchBlocked as b:
            box["rec"] = (b.status, b.reason, {"missing": list(b.missing), **b.detail})
            raise
        except ValidationFailed as v:
            box["rec"] = (FAILED_VALIDATION, v.reason, v.detail)
            raise
        except Exception as e:                                       # noqa: BLE001 — recorded, traceback written by run_stage, re-raised
            box["rec"] = (FAILED, f"{type(e).__name__}: {e}", {})
            raise
        box["ctx"] = ctx
        return out

    out = run.run_stage(step, wrapped, note=note, inputs=inputs)
    if "rec" in box:
        status, reason, detail = box["rec"]
        ledger.set(step, status, reason, detail=detail)
        return None
    ctx = box.get("ctx", {})
    rec = run.stage_records().get(step, {})
    ledger.set(step, CACHED if ctx.get("cache") == "hit" else COMPLETED, note, outputs=[str(p) for p in ctx.get("outputs", [])],
               seconds=rec.get("seconds"))
    return out


def gate_row(criterion: str, value: float | None, threshold: float | None, passed: bool | None, ci_lo: float | None = None,
             ci_hi: float | None = None, note: str = "", blocked: str | None = None) -> dict[str, Any]:
    """One gate criterion. ``passed=None`` (no computed evidence) → NOT_EVALUATED; ``blocked`` → that BLOCKED_* status."""
    if blocked is not None:
        status = blocked
    elif passed is None or value is None:
        status = NOT_EVALUATED
    else:
        status = PASS if passed else FAIL
    return {"criterion": criterion, "value": value, "ci_lo": ci_lo, "ci_hi": ci_hi, "threshold": threshold, "status": status, "note": note}


__all__ = ["BLOCKED", "BLOCKED_INSUFFICIENT_SAMPLE", "BLOCKED_MISSING_INPUT", "BLOCKED_PROVENANCE", "CACHED", "COMPLETED", "DONE", "FAIL", "FAILED",
           "FAILED_VALIDATION", "GATE_STATUSES", "NOT_EVALUATED", "NOT_RUN", "PASS", "READY", "STEP_STATUSES", "ResearchBlocked", "StatusLedger",
           "ValidationFailed", "gate_row", "run_step"]
