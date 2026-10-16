"""Stage framework of the unified notebook: gate → steps → conclusion → report, with reuse and failure isolation.

A stage object is created in the notebook right before its section (so outputs written by earlier stages count). Its
constructor evaluates the gate once:

    disabled   CFG["stages"][track] is False
    skipped    an input is missing (``data_access.REQUIREMENTS``), the warehouse is absent, a software check the track
               depends on failed, or an upstream stage failed in this run
    reused     ``reuse_completed_stages`` and the warehouse run manifest of the track says it was executed in the SAME
               run mode with the SAME ``M12Config`` hash (plus track-specific rules, e.g. 12E module definitions)
    pending    will run

Every method decorated with ``@step`` is a no-op (one explanatory line) unless the stage runs. An exception inside a
step marks the stage failed, writes the traceback to ``logs/<stage>_traceback.txt`` and — with
``continue_on_error`` — lets the notebook continue; dependent stages then gate themselves off. ``finish()`` always
writes the stage report (executed, reused, skipped or failed) and the stage record.
"""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Any, Callable, Sequence

from ..artifacts import STATUS_DONE, STATUS_FAILED, STATUS_REUSED, STATUS_SKIPPED, ResearchRun, descriptive_figure_name, stage_slug
from ..data_access import REQUIREMENTS, check_stage

# software checks each source notebook asserted (12.0 asserted all of them)
_CHECK_SETS: dict[str, Callable[[Any], tuple]] = {
    "12.0": lambda v: v.ALL_CHECKS,
    "12C": lambda v: (*v.CHECKS_GENERAL, *v.CHECKS_DISTRIBUTIONAL, *v.CHECKS_FUNCTIONAL[:2]),
    "12B": lambda v: (*v.CHECKS_GENERAL, *v.CHECKS_MEASUREMENT),
    "12A": lambda v: (*v.CHECKS_GENERAL, *v.CHECKS_FUNCTIONAL),
    "12D": lambda v: (*v.CHECKS_GENERAL, *v.CHECKS_MULTIVIEW),
    "12E": lambda v: (*v.CHECKS_GENERAL, *v.CHECKS_TRANSPORT),
    "12G": lambda v: (),
}
# warehouse table prefix of each track (outputs mirrored on reuse)
PREFIX = {"12.0": "m12_", "12C": "m12c_", "12B": "m12b_", "12A": "m12a_", "12D": "m12d_", "12E": "m12e_", "12G": "m12g_"}
FIG_KEY = {"12.0": "0", "12C": "C", "12B": "B", "12A": "A", "12D": "D", "12E": "E", "12G": "G"}


def required_check_names(track: str) -> list[str]:
    from nflbdb.research.advanced_movement.m12 import validation as v12

    return [f.__name__.removeprefix("check_") for f in _CHECK_SETS.get(track, lambda v: ())(v12)]


def run_software_checks(run: ResearchRun) -> Any:
    """``v12.run_all(v12.ALL_CHECKS)`` once (the superset of every notebook's subset) → ``diagnostics/software_checks``."""
    from nflbdb.research.advanced_movement.m12 import validation as v12

    checks = v12.run_all(v12.ALL_CHECKS)
    run.save_table(checks, "software_checks", "checks", subdir="diagnostics")
    return checks


def display_df(df: Any, rows: int = -1, width: int = 260, strlen: int = 120) -> None:
    import polars as pl

    try:
        from IPython.display import display
    except ImportError:                                        # pragma: no cover
        print(df)
        return
    if df is None:
        return
    if isinstance(df, pl.DataFrame):
        with pl.Config(tbl_rows=rows, tbl_width_chars=width, fmt_str_lengths=strlen):
            display(df)
    else:
        display(df)


def md(text: str) -> None:
    try:
        from IPython.display import Markdown, display

        display(Markdown(text))
    except ImportError:                                        # pragma: no cover
        print(text)


def run_guarded(run: ResearchRun, ucfg: Any, stage: str, fn: Callable[[], Any], enabled: bool = True, reason: str = "") -> Any:
    """Run a non-stage section (EDA, unified evaluation, final report) with the same record / failure isolation as a stage."""
    if not enabled:
        run.stage_begin(stage, "skipped")
        run.stage_end(stage, STATUS_SKIPPED, note=reason or "disabled")
        print(f"[{stage}] skipped — {reason or 'disabled'}")
        return None
    run.stage_begin(stage)
    try:
        out = fn()
    except Exception as e:                                         # noqa: BLE001
        run.stage_fail(stage, e)
        print(f"\n{'=' * 100}\n[{stage}] FAILED: {type(e).__name__}: {e}\n{'=' * 100}")
        if not ucfg.continue_on_error:
            raise
        return None
    run.stage_end(stage, STATUS_DONE)
    return out


def step(fn: Callable[..., Any]) -> Callable[..., Any]:
    @functools.wraps(fn)
    def wrapper(self: "Stage", *args: Any, **kwargs: Any) -> Any:
        if self.state != "pending" and self.state != "running":
            self._announce(fn.__name__)
            return None
        if self.state == "pending":
            self._start()
        try:
            return fn(self, *args, **kwargs)
        except Exception as e:                                 # noqa: BLE001 — recorded, re-raised unless continue_on_error
            self.state = "failed"
            self.reason = f"{type(e).__name__} in {fn.__name__}: {e}"
            self.run.stage_fail(self.track, e)
            print(f"\n{'=' * 100}\n[{self.track}] FAILED in step `{fn.__name__}`: {type(e).__name__}: {e}\n"
                  f"traceback: {self.run.rel(self.run.root / 'logs' / (stage_slug(self.track) + '_traceback.txt'))}\n"
                  f"{'dependent stages gate themselves off; the rest of the notebook continues' if self.ucfg.continue_on_error else ''}\n{'=' * 100}")
            if not self.ucfg.continue_on_error:
                raise
            return None
    return wrapper


class Stage:
    """Base of the warehouse stages. Subclasses set ``track``, ``title``, ``report_name``, ``methodology`` and define steps."""

    track: str = ""
    title: str = ""
    report_name: str = ""
    methodology: str = ""

    def __init__(self, ctx: Any, run: ResearchRun, ucfg: Any, sources: Any, checks: Any = None):
        self.ctx, self.run, self.ucfg, self.sources, self.checks = ctx, run, ucfg, sources, checks
        self.state, self.reason = "pending", ""
        self.missing_optional: list[str] = []
        self.facts: dict[str, Any] | None = None
        self.conclusion_md: str | None = None
        self.m12_status: str | None = None
        self.report_tables: list[tuple[str, Any, Sequence[str] | None]] = []
        self._announced: set[str] = set()
        self._gate()

    # ================================================================== gate
    def _gate(self) -> None:
        if not self.ucfg.enabled(self.track):
            self.state, self.reason = "disabled", "disabled in CFG['stages']"
            return
        g = check_stage(self.track, self.sources.research_root, self.ucfg.warehouse_run_mode, self.sources.bundle_dir is not None)
        if self.ctx is None or not g["ok"]:
            self.state, self.reason = "skipped", g["reason"] or "no warehouse context"
            return
        self.missing_optional = g["missing_optional"]
        if self.checks is not None and getattr(self.checks, "height", 0):
            import polars as pl

            need = set(required_check_names(self.track))
            bad = self.checks.filter(pl.col("check").is_in(list(need)) & (pl.col("status") == "FAIL"))["check"].to_list()
            if bad:
                self.state, self.reason = "skipped", f"software check(s) failed: {bad} (see diagnostics/software_checks)"
                return
        for up in REQUIREMENTS[self.track].upstream:
            if self.run.stage_status(up) == STATUS_FAILED:
                self.state, self.reason = "skipped", f"upstream stage {up} failed in this run"
                return
        if self.ucfg.reuse_completed_stages and self._reusable():
            self.state, self.reason = "reused", "executed before in this run mode with the same M12Config hash (warehouse run manifest)"

    def _reusable(self) -> bool:
        from nflbdb.research.advanced_movement.m12 import config as m12c
        from nflbdb.research.advanced_movement.m12 import reporting as rp

        m = rp.manifest(self.ctx.store, self.track)
        if not m or m.get("status") in (m12c.PENDING, m12c.INCOMPLETE, m12c.NOT_FEASIBLE, None):
            return False
        prov = m.get("provenance") or {}
        return (prov.get("run_mode") == self.ctx.controls.run_mode and prov.get("config_hash") == self.ctx.C.hash()
                and self._reuse_extra_ok(m))

    def _reuse_extra_ok(self, manifest: dict[str, Any]) -> bool:
        return True

    def explain(self) -> None:
        """One line: what this stage will do and why."""
        what = {"pending": "will RUN", "reused": "REUSES its persisted results", "skipped": "is SKIPPED", "disabled": "is DISABLED"}.get(self.state, self.state)
        extra = f" · optional inputs missing: {', '.join(self.missing_optional)}" if self.missing_optional and self.state == "pending" else ""
        md(f"**{self.track} — {self.title}** {what}{(': ' + self.reason) if self.reason else ''}{extra}")

    def _announce(self, step_name: str) -> None:
        if self.state not in self._announced:
            self._announced.add(self.state)
            print(f"[{self.track}] steps skipped — stage {self.state}: {self.reason}")

    def _start(self) -> None:
        self.state = "running"
        self.ctx.set_stage(self.track)
        self.run.stage_begin(self.track, self.title)

    @property
    def active(self) -> bool:
        return self.state in ("pending", "running")

    # ================================================================== shortcuts (valid while running)
    @property
    def C(self) -> Any:
        return self.ctx.C

    @property
    def D(self) -> Any:
        return self.ctx.D

    @property
    def store(self) -> Any:
        return self.ctx.store

    @property
    def controls(self) -> Any:
        return self.ctx.controls

    def W(self, df: Any, name: str, csv: bool = True) -> Any:
        return self.ctx.store.write_frame(df, name, csv)

    def opt(self, key: str) -> bool:
        return self.ucfg.opt(self.track, key)

    def knob(self, key: str) -> Any:
        return self.ucfg.knob(self.track, key)

    def add_report_table(self, title: str, df: Any, cols: Sequence[str] | None = None) -> None:
        if df is not None and getattr(df, "height", 0):
            self.report_tables.append((title, df, cols))

    # ================================================================== reuse: mirror persisted outputs
    def _mirror_persisted(self) -> None:
        import polars as pl

        from ..warehouse import PROCESSED

        store, run, st = self.ctx.store, self.run, self.track
        pre = PREFIX[st]
        n_t = n_f = 0
        for p in sorted(store.data_dir.glob(f"{pre}*.parquet")):
            name = p.stem
            if st == "12.0" and name.startswith(("m12a_", "m12b_", "m12c_", "m12d_", "m12e_", "m12f_", "m12g_")):
                continue
            df = pl.read_parquet(p)
            if name in PROCESSED:
                run.save_processed_dataset(df, name, st, {"warehouse_table": name, "reused": True})
            else:
                run.save_table(df, name, st, note="reused from the warehouse")
            n_t += 1
        for p in sorted(store.data_dir.glob(f"{pre}*.json")):
            if st == "12.0" and not p.name.startswith(("m12_foundation", "m12_dependency")):
                continue
            run.copy_in(p, f"manifests/research/{p.name}", st, kind="research_json")
        run.copy_in(store.data_dir / f"m12_run_{st.replace('.', '_')}.json", f"manifests/research/m12_run_{st.replace('.', '_')}.json", st, kind="research_json")
        for fmt in ("png", "svg", "pdf"):
            for p in sorted(store.figures_dir.glob(f"AM12_{FIG_KEY[st]}_*.{fmt}")):
                run.copy_in(p, f"figures/{fmt}/{descriptive_figure_name(p.stem, st)}.{fmt}", st, kind="figure")
                n_f += fmt == "png"
        for p in sorted((store.tables_dir / "plot_data").glob(f"AM12_{FIG_KEY[st]}_*.csv")):
            stem, _, part = p.stem.partition("__")
            run.copy_in(p, f"figures/data/{descriptive_figure_name(stem, st)}__{part or 'data'}.csv", st, kind="figure_data")
        c = store.reports_dir / "conclusions" / f"{st.replace('.', '_')}_CONCLUSION.md"
        if c.is_file():
            self.conclusion_md = c.read_text(encoding="utf-8")
        from nflbdb.research.advanced_movement.m12 import reporting as rp

        self.m12_status = (rp.manifest(store, st) or {}).get("status")
        self.run.log(f"reused: {n_t} tables, {n_f} figures mirrored from {store.cfg.paths.rel(store.data_dir)}")

    # ================================================================== finish / report
    def finish(self) -> Path | None:
        """Write the stage report and record. Safe to call in every state."""
        if self.state == "finished":
            return None
        rep = None
        if self.state == "reused":
            self.run.stage_begin(self.track, "reuse")
            self.ctx.set_stage(self.track)
            self._mirror_persisted()
            rep = self._write_report("reused")
            self.run.stage_end(self.track, STATUS_REUSED, note=f"m12 status {self.m12_status}")
        elif self.state == "running":
            if self.conclusion_md is None:
                self.reason = "conclusion step did not run"
                rep = self._write_report("incomplete")
                self.run.stage_end(self.track, STATUS_FAILED, note=self.reason)
            else:
                rep = self._write_report("executed")
                self.run.stage_end(self.track, STATUS_DONE, note=f"m12 status {self.m12_status}")
        elif self.state == "failed":
            rep = self._write_report("failed")
        else:
            self.run.stage_begin(self.track, self.state)
            rep = self._write_report(self.state)
            self.run.stage_end(self.track, STATUS_SKIPPED, note=self.reason)
        self.final = self.state
        self.state = "finished"
        if rep:
            md(f"Report: `{self.run.rel(rep)}`")
        return rep

    def _write_report(self, outcome: str) -> Path | None:
        from ..reporting import stage_report

        text = stage_report(self, outcome)
        paths = self.run.save_markdown(text, self.report_name, self.track)
        return paths[0] if paths else None


__all__ = ["FIG_KEY", "PREFIX", "Stage", "display_df", "md", "required_check_names", "run_guarded", "run_software_checks", "step"]
