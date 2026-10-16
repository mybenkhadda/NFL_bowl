"""12G — advanced research synthesis and metric tournament (ported from ``12g_advanced_movement_research_synthesis.ipynb``).

Which advanced methods outperform simpler physical models under identical cohorts, baselines and evaluation protocols —
and what should the next scientific milestone be? No naive meta-analysis: different targets, metrics and units are
never pooled; missing tracks are ``execution_pending`` / ``not_feasible``, never negative results.
"""

from __future__ import annotations

from .base import Stage, display_df, md, step


class Synthesis(Stage):
    track = "12G"
    title = "Advanced research synthesis and metric tournament"
    report_name = "12g_research_synthesis"
    methodology = (
        "Evidence registry (G1): every track's persisted `m12<x>_comparisons` (one schema) and `m12<x>_evidence` rows, with each track's run "
        "status. The six declared comparative experiments (G3), each with its own metric on the athletes its track used; the evidence-status "
        "matrix (G2) shows text statuses per method × category, never a combined score. Common-cohort tournament (exploratory): linked athletes "
        "with every static Combine block present and the 12C targets; ridge ladder M1 vs M1 + each track's DECLARED representation (scalar "
        "bands, sprint Vmax, fold-fitted FPCA, training-fold regression calibration, the 12D selected factor model, OT distance to the training "
        "reference, inductive F4 fold embeddings) on IDENTICAL held-out athletes; paired athlete bootstrap; Holm and BH; within-position "
        "permutation of the best block; cohort-size sensitivity. Interpretation (G5) and decision (G6): declared rules map verdicts to "
        "candidate directions (`synthesis.recommendations`).")

    def _reuse_extra_ok(self, manifest: dict) -> bool:
        """The synthesis is reused only when no track was (re)executed in this run — otherwise it must reflect the new results."""
        from ..artifacts import STATUS_DONE

        return all(self.run.stage_status(t) != STATUS_DONE for t in ("12.0", "12C", "12B", "12A", "12D", "12E", "12F"))

    @step
    def registry(self) -> None:
        """Part 3 (G1) — evidence registry and track status."""
        from nflbdb.research.advanced_movement.m12 import foundation as fd
        from nflbdb.research.advanced_movement.m12 import reporting as rp
        from nflbdb.research.advanced_movement.m12 import synthesis as sy

        self.MAN = fd.check_foundation(self.store, self.C)
        self.STATUS, self.COMPS, self.EVID = sy.collect(self.store, rp.manifest)
        self.W(self.STATUS, "m12g_track_status")
        self.W(self.COMPS, "m12g_all_comparisons")
        self.W(self.EVID, "m12g_evidence_registry")
        display_df(self.STATUS)
        display_df(self.EVID.select("method_id", "track", "n_athletes", "reliability", "stopwatch_r2", "cross_domain_stat", "delta_rmse", "delta_lo", "delta_hi", "training_regime",
                                    "evidence_tier", "compute_cost"), rows=60, strlen=60)
        self.add_report_table("Track status", self.STATUS)

    @step
    def experiments(self) -> None:
        """Parts 4–5 (G3, G2) — the six comparative experiments and the evidence-status matrix."""
        from nflbdb.research.advanced_movement.m12 import synthesis as sy

        self.EXP = sy.experiment_table(self.COMPS, self.STATUS)
        self.MAT = sy.evidence_matrix(self.EVID, self.COMPS)
        self.W(self.EXP, "m12g_experiments")
        self.W(self.MAT, "m12g_evidence_matrix")
        t = self.EXP.select("experiment", "title", "comparison", "metric", "estimate", "lo", "hi", "n_athletes", "training_regime", "primary", "verdict")
        display_df(t, strlen=80)
        self.add_report_table("The six declared experiments", t)

    @step
    def tournament(self, encoder_byf: dict | None = None) -> None:
        """Parts 6–7 — common-cohort metric tournament (exploratory). ``encoder_byf`` = {fold: athlete embedding frame} of the inductive F4
        encoder from the nflbowl section, used when the warehouse holds no ``m12f_inductive_*`` tables."""
        import polars as pl
        from nflbdb.research.advanced_movement.m11 import combine_drills as cd
        from nflbdb.research.advanced_movement.m12 import advanced_evaluation as ae
        from nflbdb.research.advanced_movement.m12 import distributional_capacity as dc
        from nflbdb.research.advanced_movement.m12 import foundation as fd
        from nflbdb.research.advanced_movement.m12 import functional_movement as fm
        from nflbdb.research.advanced_movement.m12 import measurement_error as me
        from nflbdb.research.advanced_movement.m12 import movement_transport as mt
        from nflbdb.research.advanced_movement.m12 import multiview_traits as mv
        from nflbdb.research.advanced_movement.m12 import shared_movement_encoder as se
        from nflbdb.research.advanced_movement.m12 import synthesis as sy

        C, D, store = self.C, self.D, self.store
        TOUR, TOUR2, PERM, SIZE = pl.DataFrame(), pl.DataFrame(), {}, pl.DataFrame()
        self.RES_PRIMARY = None
        if self.opt("tournament") and store.has("m12c_targets"):
            FREEZE = dc.read_freeze(store)
            BINS = FREEZE["shared_bins"]
            reg = D.m12("m12_cohort_registry")
            off, athf = D.officials(), D.athlete_features()
            POS = {int(i): p for i, p in zip(off["nfl_id"].to_list(), off["position_group"].to_list())}
            BANDS = ["forty.band_2_4_mean_accel", "forty.band_4_6_mean_accel", "forty.band_6_8_mean_accel"]
            STATIC = [*BANDS, "forty.v_max_obs", "forty.sprint_vmax"]
            tg = store.read_frame("m12c_targets", fd.NB["12C"]).select("nfl_id", "nfl.q90_shared", "nfl.game_top_speed_q75")
            data = (off.filter(pl.col("nfl_id").is_in(reg.filter(pl.col("linked"))["nfl_id"].to_list())).join(cd.wide(athf, STATIC), on="nfl_id", how="left")
                    .join(tg, on="nfl_id", how="inner").join(D.m12("m12_fold_manifest").select("nfl_id", "fold"), on="nfl_id", how="inner"))
            data = data.filter(pl.all_horizontal([pl.col(k).is_not_null() for k in STATIC])).sort("nfl_id")
            MODELS = {"M1": ae.Model(list(ae.M1_COLS)), "T_scalar": ae.Model([*ae.M1_COLS, *BANDS]), "T_sprint_vmax": ae.Model([*ae.M1_COLS, "forty.sprint_vmax"])}
            ids_c, Mc, _ = fm.combine_athlete_matrix(D.m12("m12_combine_curves"), BINS, C)
            MODELS["T_functional"] = ae.Model(list(ae.M1_COLS), factory=lambda a: fm.fpca_block(a, ids_c, Mc, C))
            long = me.attempt_long(D.attempt_features(), ["forty.v_max_obs"])
            stats = me.athlete_stats(long)
            MODELS["T_rc"] = ae.Model(list(ae.M1_COLS), factory=lambda a: me.rc_block(a, stats, long, "forty.v_max_obs", POS))
            if (store.data_dir / "m12d_selected_representation.json").is_file():
                sel = store.read_json("m12d_selected_representation", fd.NB["12D"])
                mv_ids, mv_X, _, mv_views = mv.athlete_matrix(athf, mv.view_features(athf["feature"].unique().to_list()))
                MODELS["T_multiview"] = ae.Model(list(ae.M1_COLS), factory=lambda a: mv.factor_block(sel["model"], sel["k"], mv_ids, mv_X, mv_views, POS, a, C, C.seed))
            if self.opt("ot_block"):
                GRID = mt.Grid(C, BINS)
                ids_t = data["nfl_id"].to_list()
                h = D.hist_athletes(ids_t)
                cubes = dc.build_cubes(h, POS, C)
                del h
                COMB = mt.combine_hists(D.m12("m12_combine_hist").filter(pl.col("nfl_id").is_in(ids_t)), GRID, C, "forty", "accelerating")
                NFLD = mt.nfl_hists(cubes, GRID, "accelerating")
                MODELS["T_ot"] = ae.Model(list(ae.M1_COLS), factory=lambda a: mt.ot_block(a, COMB, NFLD, POS, GRID, C))
            byf = {}
            for f in sorted(set(data["fold"].to_list())):
                nm = f"m12f_inductive_{se.PRIMARY}_f{f}"
                if store.has(nm):
                    byf[int(f)] = store.read_frame(nm, fd.NB["12F"]).drop("held_out_fold")
            if not byf and encoder_byf:
                byf = {int(f): t for f, t in encoder_byf.items()}
            self.encoder_source = "warehouse m12f_inductive tables" if byf and not encoder_byf else ("nflbowl inductive F4" if byf else "absent")
            if byf and len(byf) == len(set(data["fold"].to_list())):
                FOLD = dict(zip(data["nfl_id"].to_list(), data["fold"].to_list()))
                MODELS["T_encoder"] = ae.Model(list(ae.M1_COLS), factory=lambda a: se.embedding_block(a, FOLD, byf))
            print("tournament models:", list(MODELS), "· cohort", data.height, "athletes · encoder block:", self.encoder_source)
            tabs = []
            for tgt in ("nfl.q90_shared", "nfl.game_top_speed_q75"):
                res = ae.oof_ladder(data, tgt, MODELS, C, C.seed)
                t = ae.compare(res, [(m, "M1") for m in MODELS if m != "M1"], C, C.seed, target=tgt)
                t = ae.add_adjusted(ae.add_adjusted(t, how="holm"), how="bh")
                tabs.append(t)
                if tgt == "nfl.q90_shared":
                    self.RES_PRIMARY = res
            TOUR, TOUR2 = tabs
            best = TOUR.sort("delta_rmse", descending=True)["model"][0] if TOUR.height else None
            if best:
                PERM = ae.permutation_blocks(data, "nfl.q90_shared", MODELS["M1"], MODELS[best], C, C.seed, n_perm=min(C.n_perm, 100))
                if self.opt("cohort_size"):
                    SIZE = sy.cohort_size_sensitivity(data, "nfl.q90_shared", MODELS["M1"], MODELS[best], tuple(self.knob("size_fractions")), int(self.knob("size_reps")), C, C.seed).with_columns(pl.lit(best).alias("model"))
        elif not self.opt("tournament"):
            md("_Tournament not run (options['12G']['tournament'] is off)._")
        self.TOUR, self.TOUR2, self.PERM, self.SIZE = TOUR, TOUR2, PERM, SIZE
        self.W(TOUR, "m12g_tournament")
        self.W(TOUR2, "m12g_tournament_secondary")
        self.W(pl.DataFrame([PERM]) if PERM else pl.DataFrame(), "m12g_tournament_permutation")
        self.W(SIZE, "m12g_cohort_sensitivity")
        for t in (TOUR, TOUR2):
            display_df(t.select("target", "model", "n_athletes", "delta_rmse", "delta_lo", "delta_hi", "delta_pct", "p_boot_le0_holm", "p_boot_le0_bh", "verdict")) if t.height else md("_tournament not run (needs 12C)_")
        display_df(pl.DataFrame([PERM])) if PERM else md("_no permutation_")
        if SIZE.height:
            display_df(SIZE.group_by("fraction").agg(pl.col("delta_rmse").mean(), pl.col("delta_rmse").quantile(0.1).alias("q10"), pl.col("delta_rmse").quantile(0.9).alias("q90")).sort("fraction"))
        if TOUR.height:
            self.add_report_table("Common-cohort tournament (nfl.q90_shared; exploratory, Holm / BH)", TOUR,
                                  ["target", "model", "n_athletes", "delta_rmse", "delta_lo", "delta_hi", "p_boot_le0_holm", "p_boot_le0_bh", "verdict"])

    @step
    def interpretation(self) -> None:
        """Part 8 (G5, G6) — rule-based recommendations; answers to Q1–Q7 (counts of primary verdicts)."""
        import polars as pl
        from nflbdb.research.advanced_movement.m12 import config as m12c
        from nflbdb.research.advanced_movement.m12 import synthesis as sy

        COMPS, STATUS, TOUR = self.COMPS, self.STATUS, self.TOUR
        self.REC = sy.recommendations(STATUS, COMPS, TOUR if TOUR.height else None)
        self.W(self.REC, "m12g_recommendations")
        Q = {"Q1 functional information (12A)": 1, "Q2 measurement error (12B)": 2, "Q3 NFL movement distributions (12C)": 3, "Q4 latent traits (12D)": 4,
             "Q5 distribution matching (12E)": 5, "Q6 shared representation (12F)": 6}
        ans = []
        for q, k in Q.items():
            sub = COMPS.filter((pl.col("experiment") == k) & pl.col("primary")) if COMPS.height else COMPS
            tr_ = sy.EXPERIMENTS[k][0]
            st_ = STATUS.filter(pl.col("track") == tr_)["status"]
            ans.append({"question": q, "track_status": st_[0] if st_.len() else m12c.PENDING, "primary_comparisons": sub.height,
                        "supported": int((sub["verdict"] == "supported").sum()) if sub.height else 0, "inconclusive": int((sub["verdict"] == "inconclusive").sum()) if sub.height else 0,
                        "reversed": int((sub["verdict"] == "reversed").sum()) if sub.height else 0})
        ans.append({"question": "Q7 incremental value (12G tournament)", "track_status": m12c.EXECUTED if TOUR.height else m12c.PENDING, "primary_comparisons": TOUR.height,
                    "supported": int((TOUR["verdict"] == "model_better").sum()) if TOUR.height else 0, "inconclusive": int((TOUR["verdict"] == "inconclusive").sum()) if TOUR.height else 0,
                    "reversed": int((TOUR["verdict"] == "model_worse").sum()) if TOUR.height else 0})
        self.ANS = pl.DataFrame(ans)
        self.W(self.ANS, "m12g_question_answers")
        display_df(self.ANS, strlen=160)
        display_df(self.REC, strlen=160)
        self.add_report_table("Answers to Q1–Q7 (counts of primary verdicts)", self.ANS)
        self.add_report_table("Rule-based recommendations", self.REC)

    @step
    def figures(self) -> None:
        """Part 9 (G4) — track status, evidence matrix, primary forest, tournament, reliability vs novelty, cohort size, experiments."""
        import numpy as np
        import polars as pl
        from nflbdb.research.advanced_movement.m12 import figures as fg

        store, STATUS, MAT, COMPS, TOUR, TOUR2, EVID, SIZE, EXP = self.store, self.STATUS, self.MAT, self.COMPS, self.TOUR, self.TOUR2, self.EVID, self.SIZE, self.EXP
        st_grid = STATUS.select("track", pl.lit("status").alias("aspect"), "status")
        store.show(fg.fig_state_grid(st_grid, "track", "aspect", "status", fg.STATUS_COLORS, "Milestone 12 track status", "Status recorded by each track's run manifest in this run mode (execution_pending = not yet run).", label="status"),
                   "AM12_G_01_track_status", {"status": STATUS}, source="m12_run_*.json")
        if MAT.height:
            store.show(fg.fig_state_grid(MAT, "method_id", "category", "status", fg.STATUS_COLORS, "Evidence-status matrix",
                                         "Per method and evidence category: supported · inconclusive · rejected / reversed · descriptive · not tested. Text labels carry the meaning; categories are never added up.", label="status"),
                       "AM12_G_02_evidence_matrix", {"matrix": MAT}, publication=True, source="m12g_evidence_matrix")
        pc = COMPS.filter(pl.col("primary")).with_columns((pl.col("track") + " · " + pl.col("comparison")).alias("label")) if COMPS.height else COMPS
        if pc.height:
            store.show(fg.fig_forest(pc, "label", "estimate", "lo", "hi", "Declared primary comparisons of every track",
                                     "Estimate with 95 % interval in each comparison's own units (see the metric column); rows are not comparable across tracks and are not pooled.", "estimate (track-specific units)",
                                     color_by="track"), "AM12_G_03_primary_forest", {"primary": pc}, publication=True, source="m12g_all_comparisons")
        if TOUR.height:
            tf = pl.concat([TOUR, TOUR2], how="diagonal_relaxed").with_columns((pl.col("target") + " · " + pl.col("model") + " (n=" + pl.col("n_athletes").cast(pl.String) + ")").alias("label"))
            store.show(fg.fig_forest(tf, "label", "delta_rmse", "delta_lo", "delta_hi", "Common-cohort tournament: paired improvement over official tests",
                                     "Δ RMSE = RMSE(M1) − RMSE(M1 + block) on IDENTICAL held-out athletes per target, paired athlete bootstrap 95 % CI; exploratory (Holm / BH in the table).", "Δ RMSE (target units)",
                                     color_by="target"), "AM12_G_04_tournament", {"tournament": tf}, publication=True, source="m12g_tournament")
        rn = EVID.filter(pl.col("reliability").is_not_null()).with_columns((1 - pl.col("stopwatch_r2").fill_null(np.nan)).alias("novelty"), pl.col("track").alias("position_group"))
        if rn.filter(pl.col("novelty").is_not_nan()).height:
            store.show(fg.fig_scatter(rn.filter(pl.col("novelty").is_not_nan()), "reliability", "novelty", "Reliability versus novelty",
                                      "Each method: measurement reliability (track-specific: ICC, split-half or test–retest) against 1 − stopwatch R² (information not already in official tests). Colour = track.",
                                      "reliability", "1 − stopwatch R²"), "AM12_G_05_reliability_vs_novelty", {"evidence": rn}, publication=True, source="m12g_evidence_registry")
        if SIZE.height:
            sz = SIZE.group_by("fraction").agg(pl.col("delta_rmse").mean().alias("mean"), pl.col("delta_rmse").quantile(0.1).alias("lo"), pl.col("delta_rmse").quantile(0.9).alias("hi"))
            store.show(fg.fig_lines(sz, "fraction", "mean", None, "Cohort-size sensitivity of the best tournament block",
                                    f"Δ RMSE of {SIZE['model'][0]} vs M1 on random athlete subsamples (10th–90th percentile band, {self.knob('size_reps')} subsamples per fraction).",
                                    "share of the common cohort", "Δ RMSE", "lo", "hi", ref=0.0), "AM12_G_06_cohort_size", {"size": SIZE}, source="m12g_cohort_sensitivity")
        if EXP.height:
            ex = EXP.filter(pl.col("estimate").is_not_null()).with_columns((pl.col("experiment").cast(pl.String) + " · " + pl.col("comparison")).alias("label"))
            if ex.height:
                store.show(fg.fig_forest(ex.head(60), "label", "estimate", "lo", "hi", "The six comparative experiments",
                                         "All comparisons of experiments 1–6 as persisted by their tracks (own units per row; ≤ 60 rows shown, the table holds all).", "estimate", color_by="track"),
                           "AM12_G_07_experiments", {"experiments": ex}, source="m12g_experiments")

    @step
    def conclude(self) -> None:
        """Parts 10–12 — Milestone 12 documents, reproducibility record, conclusion, run manifest."""
        import polars as pl
        from nflbdb.research.advanced_movement.m12 import config as m12c
        from nflbdb.research.advanced_movement.m12 import foundation as fd
        from nflbdb.research.advanced_movement.m12 import reporting as rp

        store, STATUS, EXP, TOUR, TOUR2, COMPS, REC, ANS, MAN = self.store, self.STATUS, self.EXP, self.TOUR, self.TOUR2, self.COMPS, self.REC, self.ANS, self.MAN
        nl = chr(10)
        rp.update_doc(store, "ADVANCED_METHOD_COMPARISON.md", "### Track status" + nl + nl + rp.md_table(STATUS) + nl + "### Experiments 1–6" + nl + nl
                      + rp.md_table(EXP, ["experiment", "title", "comparison", "metric", "estimate", "lo", "hi", "n_athletes", "training_regime", "verdict"], max_rows=80)
                      + nl + "### Common-cohort tournament (exploratory)" + nl + nl + rp.md_table(TOUR, ["target", "model", "n_athletes", "delta_rmse", "delta_lo", "delta_hi", "p_boot_le0_holm", "p_boot_le0_bh", "verdict"])
                      + nl + nl + rp.md_table(TOUR2, ["target", "model", "n_athletes", "delta_rmse", "delta_lo", "delta_hi", "verdict"]))
        neg = COMPS.filter(pl.col("primary") & pl.col("verdict").is_in(["inconclusive", "reversed", "insufficient_athletes_for_interval"])) if COMPS.height else COMPS
        rp.update_doc(store, "NEGATIVE_RESULTS.md", "### Primary comparisons that did not support their hypothesis" + nl + nl + rp.md_table(neg, ["track", "comparison", "metric", "estimate", "lo", "hi", "n_athletes", "verdict"])
                      + nl + "### Tournament blocks without an interval above 0" + nl + nl + rp.md_table(TOUR.filter(pl.col("verdict") != "model_better") if TOUR.height else TOUR, ["target", "model", "delta_rmse", "delta_lo", "delta_hi", "verdict"])
                      + nl + "### Not run / not feasible" + nl + nl + rp.md_table(STATUS.filter(pl.col("status").is_in([m12c.PENDING, m12c.NOT_FEASIBLE, m12c.INCOMPLETE]))))
        rp.update_doc(store, "NEXT_RESEARCH_DIRECTIONS.md", "### Rule-based recommendations (synthesis.recommendations)" + nl + nl + rp.md_table(REC) + nl + "### Questions Q1–Q7" + nl + nl + rp.md_table(ANS))
        rp.update_doc(store, "RESEARCH_OBJECTIVES.md", "### Answers so far (counts of primary verdicts)" + nl + nl + rp.md_table(ANS))
        rp.update_doc(store, "EXPERIMENT_REGISTRY.md", "### Execution status" + nl + nl + rp.md_table(rp.status_table(store)))
        repro = rp.status_table(store)
        fz = store.read_json("m12c_target_freeze", fd.NB["12C"]) if (store.data_dir / "m12c_target_freeze.json").is_file() else {}
        rp.update_doc(store, "REPRODUCIBILITY.md", "### Run manifests (this run mode)" + nl + nl + rp.md_table(repro) + nl
                      + f"### Signatures{nl}{nl}- foundation config `{MAN.get('config_hash')}` · cohort `{MAN.get('cohort_signature')}` · folds `{MAN.get('fold_signature')}` · grids `{MAN.get('grid_signature')}`{nl}"
                      + f"- 12C target freeze `{fz.get('freeze_hash', '—')}` ({fz.get('estimator', '—')}, {fz.get('frozen_at', '—')}){nl}- seed {self.controls.seed} · run mode {self.controls.run_mode}{nl}")
        self.W(repro, "m12g_reproducibility")
        status = m12c.EXECUTED if TOUR.height else m12c.INCOMPLETE
        n_t = int(self.RES_PRIMARY.y.size) if (TOUR.height and self.RES_PRIMARY is not None) else 0
        facts = {"status": status, "run_mode": self.controls.run_mode,
                 "sample": f"tracks executed: {', '.join(STATUS.filter(~pl.col('status').is_in([m12c.PENDING, m12c.NOT_FEASIBLE]))['track'].to_list()) or 'none'}; tournament cohort {n_t} athletes.",
                 "evidence": [f"{r['question']}: {r['supported']} supported / {r['inconclusive']} inconclusive / {r['reversed']} reversed of {r['primary_comparisons']} (track {r['track_status']})" for r in ANS.iter_rows(named=True)]
                             + [f"recommendation: {r['direction']} — {r['triggered_by']} ({r['strength']})" for r in REC.iter_rows(named=True)],
                 "alternatives": ["Tracks use different targets and cohorts by design; only the tournament compares methods on identical athletes."],
                 "relevance": "Turns six method families into one evidence-based choice of what to pursue, stop or confirm next.",
                 "limitations": ["One draft cohort; every gain is exploratory until an independent class confirms it.", "Tournament blocks are the declared representations of each track, not tuned for the tournament."],
                 "next": [r["direction"] for r in REC.iter_rows(named=True)]}
        text = rp.conclusion_markdown("12G", facts)
        md(text)
        rp.write_conclusion(store, "12G", text)
        rp.run_manifest(store, "12G", self.ctx.provenance("12G"), ["m12g_track_status", "m12g_evidence_registry", "m12g_experiments", "m12g_tournament", "m12g_recommendations", "m12g_question_answers"], status)
        self.facts, self.conclusion_md, self.m12_status = facts, text, status

    def run_all(self, encoder_byf: dict | None = None) -> None:
        self.registry()
        self.experiments()
        self.tournament(encoder_byf)
        self.interpretation()
        self.figures()
        self.conclude()


__all__ = ["Synthesis"]
