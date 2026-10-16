"""12.0 — advanced movement research foundation (ported from ``12_00_advanced_movement_research_foundation.ipynb``).

Inventory, not a test: the Combine curve tables on the NFL histogram edges, the shared speed range (support counts
only), the cohort registry, the fold manifest and the feasibility of every track. Later stages refuse other grids
(``fd.check_foundation``).
"""

from __future__ import annotations

from typing import Any

from .base import Stage, display_df, md, step


class Foundation(Stage):
    track = "12.0"
    title = "Research foundation and feasibility"
    report_name = "12_0_foundation"
    methodology = (
        "Reuses the Milestone 11 foundation (officials, attempt index / features, NFL targets, athlete folds, envelopes) and the NFL "
        "acceleration–speed histograms `m11_envelope_hist` (athlete × game × 0.5 yd/s × 0.25 yd/s² frame counts; scanned lazily). The only new "
        "pass is over the Combine frames (one drill family at a time; quality `reject` excluded; longest valid segment per attempt) to build "
        "per-attempt rising-phase a(v) curves (interpolated at bin mid-speeds, rule `rising_phase_interp_v2`), speed–time curves from onset and "
        "histograms on the SAME edges as the NFL histograms. The shared speed range is the longest contiguous run of 0.5 yd/s bins in which at "
        "least `shared_support_share` of linked athletes are supported in BOTH domains — decided by support counts only, never by an outcome. "
        "Folds are the Milestone 11 athlete folds.")

    @step
    def inventory(self) -> None:
        """Part 3 — memory plan and athlete / attempt / histogram counts."""
        import polars as pl
        from nflbdb.research.advanced_movement.m12 import advanced_evaluation as ae

        D = self.D
        self.off, self.folds, self.aidx, self.env = D.officials(), D.folds(), D.attempt_index(), D.envelopes()
        self.hist = D.hist()
        hist_rows = int(self.hist.select(pl.len()).collect().item())
        hist_mb = ae.file_mb(self.store.discovery.data_dir / "m11_envelope_hist.parquet")
        plan = ae.memory_plan([
            {"item": "m11_envelope_hist (lazy; aggregated only)", "n_rows": hist_rows, "n_cols": 6, "bytes_per_value": 4, "on_disk_mb": hist_mb,
             "note": "never materialised in 12.0"},
            {"item": "Combine frames of the largest family (one at a time)", "n_rows": int(self.aidx["n_frames"].sum()), "n_cols": 25, "bytes_per_value": 8,
             "note": "upper bound: all families together"},
            {"item": "m12_combine_hist (attempt × speed × accel cells)", "n_rows": int(self.aidx.height * 120), "n_cols": 6, "bytes_per_value": 8, "note": "estimate"}],
            self.knob("memory_budget_gb"))
        display_df(plan)
        self.run.save_table(plan, "12_0_memory_plan", self.track, subdir="diagnostics")
        print(f"{self.off.height} Combine registry athletes · {self.aidx.height} tracked attempts · {hist_rows:,} NFL histogram cells")

    @step
    def curve_tables(self) -> None:
        """Part 3 — Combine curve tables (restored when persisted), NFL bin support, game top speed, tracking quality."""
        import polars as pl
        from nflbdb.research.advanced_movement.m11 import sources as src
        from nflbdb.research.advanced_movement.m12 import foundation as fd

        names = ["m12_combine_curves", "m12_combine_speed_time", "m12_combine_hist", "m12_combine_quality"]
        force = self.opt("force_rebuild_curves")
        if all(self.store.has(n) for n in names) and not force:
            comb = {n: self.store.read_frame(n, fd.NB["12.0"]) for n in names}
            print("Combine curve tables restored from this run mode")
            for n, d in comb.items():                                   # mirror the restored inputs into RUN_DIR (no warehouse rewrite)
                self.run.save_processed_dataset(d, n, self.track, {"warehouse_table": n, "restored": True})
        else:
            comb = fd.combine_tables(self.ctx.S, self.C, list(src.FAMILY_ORDER))
            for n, d in comb.items():
                self.W(d, n, csv=n != "m12_combine_hist")
        self.curves, self.st, self.chist, self.qual = (comb[n] for n in names)
        self.nsup = fd.nfl_bin_support(self.hist)
        self.nsum = fd.nfl_athlete_summary(self.hist)
        self.gts = fd.game_top_speed(self.D.episodes())
        self.W(self.nsup, "m12_nfl_bin_support")
        self.W(self.gts, "m12_game_top_speed")
        q = self.qual
        self.qsum = q.group_by("family").agg(pl.len().alias("attempts"), pl.col("nfl_id").n_unique().alias("athletes"), pl.col("median_dt").median(),
                                             pl.col("irregular_share").mean(), pl.col("onset_censored").mean().alias("onset_censored_share"),
                                             pl.col("peak_at_last_frame").mean().alias("peak_at_last_frame_share"), pl.col("duration_s").median()).sort("family")
        self.W(self.qsum, "m12_tracking_quality")
        md("**Tracking quality per drill family** (sampling: median Δt; censoring: onset outside the recording; observation length: median duration).")
        display_df(self.qsum)
        self.add_report_table("Tracking quality per drill family", self.qsum)

    @step
    def definitions(self) -> None:
        """Part 4 — shared speed range, grids, cohort registry, availability, fold manifest, foundation manifest."""
        import polars as pl
        from nflbdb.research.advanced_movement.m12 import advanced_evaluation as ae
        from nflbdb.research.advanced_movement.m12 import foundation as fd
        from nflbdb.research.advanced_movement.m12 import shared_movement_encoder as se

        C = self.C
        self.linked = sorted(set(self.off["nfl_id"].to_list()) & set(self.nsum["nfl_id"].to_list()) & set(self.folds["nfl_id"].to_list()))
        self.sup_tab, self.shared = fd.shared_speed_support(self.nsup, self.curves, self.linked, C)
        self.grids = fd.movement_grids(C, self.shared)
        self.reg = fd.cohort_registry(self.off, self.folds, self.aidx, self.curves, self.nsum, self.nsup, self.env, self.gts, self.shared, C)
        self.avail = fd.availability(self.reg, self.shared, se.TORCH, C)
        fman = self.folds.with_columns(pl.lit(ae.fold_signature(self.folds)).alias("fold_signature"), pl.lit(C.hash()).alias("config_hash"))
        self.W(self.sup_tab, "m12_shared_speed_support")
        self.W(self.grids, "m12_movement_grids")
        self.W(self.reg, "m12_cohort_registry")
        self.W(self.avail, "m12_experiment_availability")
        self.W(fman, "m12_fold_manifest")
        if self.controls.save_tables:
            self.store.write_json("m12_dependency_manifest", fd.dependency_manifest())
            self.store.write_json("m12_foundation_manifest", fd.foundation_manifest(C, self.reg, self.folds, self.grids, self.shared))
        sh = self.shared
        print(f"linked athletes: {len(self.linked)} · shared speed range: {sh.get('lo')} – {sh.get('hi')} yd/s ({len(sh.get('bins', []))} bins) · rule: {sh.get('rule')}")
        display_df(self.grids, strlen=80)
        display_df(self.avail, strlen=80)
        self.add_report_table("Movement grids", self.grids, ["grid_id", "variable", "unit", "lo", "hi", "step", "n_points"])
        self.add_report_table("Experiment availability", self.avail, ["track", "eligible_athletes", "linked_athletes", "feasibility", "missing_variables"])

    @step
    def baselines(self) -> None:
        """Parts 5–8 — declared baselines and outcomes per track; position support (athletes, never frames)."""
        import polars as pl
        from nflbdb.research.advanced_movement.m12 import config as m12c

        display_df(pl.DataFrame(m12c.track_table()).select("track", "primary_outcome", "primary_comparison", "evaluation_target", "depends_on"), strlen=200)
        reg = self.reg
        self.pos_counts = (reg.filter(pl.col("in_combine")).unpivot(index=["nfl_id", "position_group"], on=[c for c in reg.columns if c.startswith("elig_")],
                                                                     variable_name="track", value_name="eligible")
                           .filter(pl.col("eligible")).group_by("position_group", "track").len().rename({"len": "n_athletes"}))
        self.W(self.pos_counts, "m12_position_support")
        piv = self.pos_counts.pivot(on="track", index="position_group", values="n_athletes").sort("position_group")
        display_df(piv)
        self.add_report_table("Eligible athletes per position and track", piv)

    @step
    def figures(self) -> None:
        """Part 9 — eligibility funnel, repeat availability, position support, cross-domain speed support, feasibility matrix."""
        import numpy as np
        import polars as pl
        from nflbdb.research.advanced_movement.m12 import figures as fg

        reg, C, store = self.reg, self.C, self.store
        funnel = pl.DataFrame([
            {"stage": "Combine registry athletes", "n_athletes": int(reg["in_combine"].sum())},
            {"stage": "with a usable forty attempt", "n_athletes": int(reg["has_forty"].sum())},
            {"stage": "with NFL tracking histograms", "n_athletes": int((reg["in_combine"] & reg["in_nfl"]).sum())},
            {"stage": "linked with an athlete fold", "n_athletes": int(reg["linked"].sum())},
            {"stage": f"12C transfer (≥ {C.min_games_athlete} games + forty)", "n_athletes": int(reg["elig_12C_transfer"].sum())},
            {"stage": "12A functional (shared speed range in both domains)", "n_athletes": int(reg["elig_12A"].sum())}])
        self.funnel = funnel
        store.show(fg.fig_funnel(funnel, "Milestone 12 athlete eligibility funnel", "Athletes (the independent unit) at each eligibility stage; later tracks use their own declared cohorts. Stage rules use support and linkage only, never an outcome value."),
                   "AM12_0_01_eligibility_funnel", {"funnel": funnel}, publication=True, source="m12_cohort_registry", n=f"{int(reg['in_combine'].sum())} athletes")
        rep_ = (self.aidx.group_by("nfl_id", "family").len().with_columns(pl.when(pl.col("len") >= 3).then(pl.lit("3+")).otherwise(pl.col("len").cast(pl.String)).alias("attempts"))
                .group_by("family", "attempts").len().rename({"len": "n_athletes"}))
        store.show(fg.fig_heatmap(rep_, "family", "attempts", "n_athletes", "Repeat-attempt availability by drill family", "Athletes by number of tracked attempts per drill family (repeats are what 12B / 12D reliability rests on)."),
                   "AM12_0_02_repeat_availability", {"repeats": rep_}, source="m11_attempt_index")
        store.show(fg.fig_heatmap(self.pos_counts, "position_group", "track", "n_athletes", "Position support per track", "Eligible athletes per position and track (declared eligibility flags of m12_cohort_registry)."),
                   "AM12_0_03_position_support", {"position_support": self.pos_counts}, source="m12_cohort_registry")
        sl = self.sup_tab.with_columns(((pl.col("speed_lo") + pl.col("speed_hi")) / 2).alias("speed")).unpivot(index="speed", on=["share_nfl", "share_combine", "share_both"],
                                                                                                             variable_name="domain", value_name="share")
        sh = self.shared
        store.show(fg.fig_lines(sl, "speed", "share", "domain", "Cross-domain speed support",
                                f"Share of the {len(self.linked)} linked athletes supported in each 0.5 yd/s speed bin (NFL, Combine rising phase, both). Shaded: the declared shared speed range ({sh.get('rule')}).",
                                "speed [yd/s]", "share of linked athletes", ref=C.shared_support_share, shade=(sh.get("lo", np.nan), sh.get("hi", np.nan))),
                   "AM12_0_04_cross_domain_speed_support", {"support": self.sup_tab}, publication=True, source="m12_shared_speed_support + m12_combine_curves",
                   n=f"{len(self.linked)} linked athletes")
        fm_ = self.avail.select("track", pl.lit("feasibility").alias("aspect"), "feasibility",
                                (pl.col("feasibility") + " · n=" + pl.col("eligible_athletes").cast(pl.String)).alias("label"))
        store.show(fg.fig_state_grid(fm_, "track", "aspect", "feasibility", fg.STATUS_COLORS, "Research feasibility matrix",
                                     "Declared feasibility per track (ready · gated on an upstream notebook · limited support · blocked) with eligible athletes. Colour repeats the text label.",
                                     label="label"),
                   "AM12_0_05_feasibility_matrix", {"availability": self.avail}, publication=True, source="m12_experiment_availability")

    @step
    def conclude(self) -> None:
        """Parts 10–12 — conclusion from executed tables, registry document, run manifest."""
        import polars as pl
        from nflbdb.research.advanced_movement.m12 import config as m12c
        from nflbdb.research.advanced_movement.m12 import reporting as rp

        reg, avail, sh = self.reg, self.avail, self.shared
        status = m12c.EXECUTED
        facts = {"status": status, "run_mode": self.controls.run_mode,
                 "sample": f"{int(reg['in_combine'].sum())} Combine athletes, {self.aidx.height} tracked attempts; {self.nsum.height} NFL athletes in the histograms; {len(self.linked)} linked.",
                 "evidence": [f"shared speed range {sh.get('lo')}–{sh.get('hi')} yd/s ({len(sh.get('bins', []))} bins)"]
                             + [f"{r['track']}: {r['eligible_athletes']} eligible athletes, feasibility {r['feasibility']}" + (f" ({r['missing_variables']})" if r["missing_variables"] else "")
                                for r in avail.iter_rows(named=True)],
                 "alternatives": ["Drill participation and NFL exposure depend on position and role: eligibility counts describe who was observed, not prospects in general."],
                 "relevance": "Fixes, before any advanced model runs, which athletes and speed states each method may use.",
                 "limitations": ["NFL histograms carry game and season, not down / alignment / role.", "The shared-range rule is a declared support threshold, not a measurement."],
                 "next": ["Run 12C (distributional NFL capacity) — the highest-priority track."]}
        text = rp.conclusion_markdown("12.0", facts)
        md(text)
        rp.write_conclusion(self.store, "12.0", text)
        nl = chr(10)
        block = ("### Track registry (declared before execution)" + nl + nl + rp.md_table(pl.DataFrame(m12c.track_table()), ["track", "primary_outcome", "primary_comparison", "evaluation_target"])
                 + nl + "### Feasibility (12.0, executed)" + nl + nl + rp.md_table(avail, ["track", "eligible_athletes", "linked_athletes", "supported_positions", "feasibility", "missing_variables"]))
        rp.update_doc(self.store, "EXPERIMENT_REGISTRY.md", block)
        out = ["m12_combine_curves", "m12_combine_speed_time", "m12_combine_hist", "m12_combine_quality", "m12_nfl_bin_support", "m12_game_top_speed", "m12_cohort_registry",
               "m12_movement_grids", "m12_shared_speed_support", "m12_fold_manifest", "m12_experiment_availability", "m12_tracking_quality", "m12_position_support"]
        rp.run_manifest(self.store, "12.0", self.ctx.provenance("12.0"), out, status, {"shared_speed": sh, "n_linked": len(self.linked)})
        self.facts, self.conclusion_md, self.m12_status = facts, text, status

    def run_all(self) -> None:
        self.inventory()
        self.curve_tables()
        self.definitions()
        self.baselines()
        self.figures()
        self.conclude()


__all__ = ["Foundation"]
