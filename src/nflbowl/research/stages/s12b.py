"""12B — measurement-error recovery and latent traits (ported from ``12b_measurement_error_and_latent_traits.ipynb``).

Are Combine → NFL relationships attenuated because tracked Combine predictors are measured unreliably? Four estimands,
never substituted for each other: observed association · corrected latent effect (RC, SIMEX, joint EM;
disattenuated r) · compatible latent range · out-of-sample value (held-out Δ RMSE of M1 + x̄ and M1 + E[θ | x̄]).
"""

from __future__ import annotations

from .base import Stage, display_df, md, step


class MeasurementError(Stage):
    track = "12B"
    title = "Measurement error and latent traits"
    report_name = "12b_measurement_error"
    methodology = (
        "Attempt-level values of the preregistered features (Milestone 11 attempt features; `forty.time_4_8` = t_to_8 − t_to_4 per attempt). "
        "Athlete value = mean of attempts (frozen Milestone 04 rule). Variance components by unbalanced method of moments (attempts nested in "
        "athletes): ICC(1), Spearman–Brown reliability of the mean of k attempts, by position and onset-recorded subset. Corrections per "
        "preregistered feature × matched NFL target: B0 naive slope · B1 regression calibration (partial reliability after M1) · B2 SIMEX "
        "(λ grid, quadratic extrapolation to −1) · B3 joint EM latent regression · disattenuated partial r · compatible latent range (observed "
        "interval ÷ √ reliabilities) · minimum detectable latent r. Held-out: M1 vs M1 + x̄ (B0) vs M1 + E[θ | x̄] (B1; μ, σ²_θ, σ²_ε from TRAINING "
        "athletes, within position when ≥ 20) on identical athletes. Assumptions stated, not tested: classical additive, nondifferential, "
        "homoscedastic within position, exchangeable attempts; with one attempt σ²_ε is not identifiable (flagged, never borrowed). Synthetic "
        "recovery grid (planted ICC × attempts × β, correlated errors, MNAR repeats, small cohort) describes estimators only.")

    @step
    def prepare(self) -> None:
        """Parts 3–4 — attempt values, athlete statistics, NFL targets (+ 12C frozen target when present), target reliability."""
        import polars as pl
        from nflbdb.research.advanced_movement.m11 import speed_state as ss
        from nflbdb.research.advanced_movement.m12 import advanced_evaluation as ae
        from nflbdb.research.advanced_movement.m12 import foundation as fd
        from nflbdb.research.advanced_movement.m12 import measurement_error as me

        D, store = self.D, self.store
        fd.check_foundation(store, self.C)
        self.off = D.officials()
        self.POS = dict(zip(self.off["nfl_id"].to_list(), self.off["position_group"].to_list()))
        self.long = me.attempt_long(D.attempt_features())
        self.stats = me.athlete_stats(self.long)
        nw = ss.nfl_wide(D.nfl_targets())
        if store.has("m12c_targets"):
            nw = nw.join(store.read_frame("m12c_targets", fd.NB["12C"]).select("nfl_id", "nfl.q90_shared"), on="nfl_id", how="left")
        self.nw = nw
        self.REL_Y = {r["target_id"]: r["reliability"] for r in D.nfl_target_reliability().iter_rows(named=True)}
        self.fman = D.m12("m12_fold_manifest").select("nfl_id", "fold")
        display_df(ae.memory_plan([{"item": "attempt values (preregistered)", "n_rows": self.long.height, "n_cols": 5}], self.knob("memory_budget_gb")))
        avail = (self.long.group_by("feature").agg(pl.col("nfl_id").n_unique().alias("athletes"), pl.len().alias("attempts"))
                 .join(self.stats.filter(pl.col("n_att") >= 2).group_by("feature").len().rename({"len": "athletes_with_repeat"}), on="feature", how="left"))
        display_df(avail.sort("feature"))
        prereg = pl.DataFrame([{"feature": p.feature, "target": p.target, "expected_sign": p.sign, "note": p.note, "nfl_reliability": self.REL_Y.get(p.target)} for p in me.PREREGISTERED])
        display_df(prereg)
        self.W(avail, "m12b_feature_availability")
        self.add_report_table("Preregistered features × matched NFL targets", prereg)

    @step
    def reliability(self) -> None:
        """Part 5 (B2) — variance components, ICC, position, attempt count, censoring."""
        import polars as pl
        from nflbdb.research.advanced_movement.m12 import figures as fg
        from nflbdb.research.advanced_movement.m12 import measurement_error as me

        C, store = self.C, self.store
        self.rt = me.reliability_table(self.long, self.POS, C, C.seed)
        self.W(self.rt, "m12b_reliability")
        t = self.rt.filter(pl.col("group").is_in(["ALL", "ALL|onset_recorded"])).select("feature", "group", "n_athletes", "n_repeat", "sigma2_theta", "sigma2_eps", "icc", "icc_lo", "icc_hi",
                                                                                   "reliability_mean", "sb_1", "sb_2", "sb_3", "identifiable")
        display_df(t)
        self.add_report_table("Repeat-attempt reliability (ICC, Spearman–Brown)", t)
        self.ra = ra = self.rt.filter(pl.col("group") == "ALL")
        store.show(fg.fig_forest(ra.sort("icc"), "feature", "icc", "icc_lo", "icc_hi", "Repeat-attempt reliability of the preregistered features",
                                 "ICC(1) by unbalanced method of moments (attempts nested in athletes), athlete-bootstrap 95 % CI.", "ICC"), "AM12_B_01_reliability", {"reliability": ra},
                   publication=True, source="m12b_reliability")
        vs = ra.with_columns((pl.col("sigma2_theta") / (pl.col("sigma2_theta") + pl.col("sigma2_eps"))).alias("between_share"),
                             (pl.col("sigma2_eps") / (pl.col("sigma2_theta") + pl.col("sigma2_eps"))).alias("within_share"))
        store.show(fg.fig_stacked(vs, "feature", ["between_share", "within_share"], "Noise versus between-athlete variance",
                                  "Share of one attempt’s variance that is between athletes (signal) and within athletes (attempt-to-attempt noise).", "share of variance"),
                   "AM12_B_02_noise_vs_signal", {"variance": vs}, source="m12b_reliability")
        rp_ = self.rt.filter(~pl.col("group").str.starts_with("ALL"))
        store.show(fg.fig_heatmap(rp_, "feature", "group", "icc", "ICC by position", "Within-position ICC (positions with ≥ 10 athletes; blank = not estimable).", fmt="{:.2f}"),
                   "AM12_B_03_icc_by_position", {"icc_position": rp_}, source="m12b_reliability")
        sb = ra.unpivot(index="feature", on=["sb_1", "sb_2", "sb_3", "sb_4"], variable_name="k", value_name="reliability").with_columns(pl.col("k").str.slice(3).cast(pl.Int64))
        store.show(fg.fig_lines(sb, "k", "reliability", "feature", "Measurement attenuation: reliability of the athlete mean by attempts",
                                "Spearman–Brown reliability of the mean of k attempts from the pooled ICC; the observed correlation is attenuated by √(reliability).",
                                "attempts averaged (k)", "reliability of the athlete mean"), "AM12_B_04_attenuation_by_attempts", {"sb": sb}, source="m12b_reliability")

    @step
    def corrections(self) -> None:
        """Parts 6–7 — B0–B3 estimands on real data and the held-out B0 / B1 ladder on identical athletes."""
        import numpy as np
        import polars as pl
        from nflbdb.research.advanced_movement.m12 import advanced_evaluation as ae
        from nflbdb.research.advanced_movement.m12 import measurement_error as me

        C, off, nw, stats, long, fman, POS = self.C, self.off, self.nw, self.stats, self.long, self.fman, self.POS
        est_rows, held_rows, oofs = [], [], []
        targets = [(p, p.target) for p in me.PREREGISTERED] + ([(p, "nfl.q90_shared") for p in me.PREREGISTERED if p.feature.startswith("forty.")] if "nfl.q90_shared" in nw.columns else [])
        for p, tgt in targets:
            if tgt not in nw.columns:
                est_rows.append({"feature": p.feature, "target": tgt, "status": "target_not_available"})
                continue
            st_f = stats.filter(pl.col("feature") == p.feature).select("nfl_id", "xbar", "n_att")
            d = off.join(st_f, on="nfl_id", how="inner").join(nw.select("nfl_id", tgt), on="nfl_id", how="inner").join(fman, on="nfl_id", how="inner").filter(pl.col(tgt).is_not_null())
            lf = long.filter(pl.col("feature") == p.feature)
            comp = me.components(lf["value"].to_numpy(), lf["nfl_id"].to_numpy())
            rel_y = self.REL_Y.get(tgt, np.nan) if tgt != "nfl.q90_shared" else np.nan
            e = me.estimands(d, "xbar", tgt, "n_att", comp["sigma2_eps"] if comp["identifiable"] else np.nan, list(ae.M1_COLS), C, C.seed, rel_y=rel_y)
            em = me.em_estimand(d, long, p.feature, tgt, list(ae.M1_COLS), C, C.seed) if comp["identifiable"] else {"status": "not_identifiable"}
            k = len(ae.M1_COLS) + max(0, d["position_group"].n_unique() - 1)
            cmp_ = me.compatible_latent(e.get("r_partial_lo", np.nan), e.get("r_partial_hi", np.nan), e.get("rel_x_partial", np.nan), rel_y, d.height, k)
            est_rows.append({"feature": p.feature, "target": tgt, "expected_sign": p.sign, "icc": comp["icc"], "identifiable": comp["identifiable"], "rel_y": rel_y,
                             **{kk: v for kk, v in e.items() if kk != "status"}, "status": e.get("status"), **{kk: v for kk, v in em.items() if kk not in ("status", "n_athletes")}, **cmp_})
            if comp["identifiable"] and d.height >= C.min_athletes:
                mods = {"M1": ae.Model(list(ae.M1_COLS)), "B0": ae.Model([*ae.M1_COLS, "xbar"]),
                        "B1": ae.Model(list(ae.M1_COLS), factory=lambda ids, _f=p.feature: me.rc_block(ids, stats, long, _f, POS))}
                res = ae.oof_ladder(d, tgt, mods, C, C.seed)
                held_rows.append(ae.compare(res, [("B0", "M1"), ("B1", "M1"), ("B1", "B0")], C, C.seed, feature=p.feature, target=tgt))
                oofs.append(res.oof_frame(feature=p.feature, target=tgt))
        self.EST = pl.DataFrame(est_rows, infer_schema_length=None)
        self.HELD = pl.concat([h for h in held_rows if h.height], how="diagonal_relaxed") if held_rows else pl.DataFrame()
        self.OOF = pl.concat(oofs, how="diagonal_relaxed") if oofs else pl.DataFrame()
        self.W(self.EST, "m12b_estimands")
        self.W(self.HELD, "m12b_heldout")
        self.W(self.OOF, "m12b_heldout_oof")
        cols = [c for c in ("feature", "target", "n_athletes", "icc", "rel_x_partial", "beta_obs", "beta_obs_lo", "beta_obs_hi", "beta_rc", "beta_rc_lo", "beta_rc_hi",
                            "beta_simex", "beta_simex_lo", "beta_simex_hi", "beta_em", "beta_em_lo", "beta_em_hi", "r_partial", "r_disattenuated", "latent_lo", "latent_hi", "mde_latent", "status")
                if c in self.EST.columns]
        display_df(self.EST.select(cols), width=300)
        held = self.HELD.select("feature", "target", "model", "reference", "n_athletes", "delta_rmse", "delta_lo", "delta_hi", "verdict") if self.HELD.height else self.HELD
        display_df(held, width=300)
        self.add_report_table("Estimands (observed, corrected, compatible latent range)", self.EST, cols)
        self.add_report_table("Held-out value (identical athletes)", held)

    @step
    def synthetic(self) -> None:
        """Part 8a (B4 grid) — planted-signal recovery, checkpointed per scenario (immutable shards; resumable)."""
        import polars as pl
        from nflbdb.research.advanced_movement.m11 import config as m11c
        from nflbdb.research.advanced_movement.m12 import advanced_evaluation as ae
        from nflbdb.research.advanced_movement.m12 import measurement_error as me

        C = self.C
        SIM, SIMS = pl.DataFrame(), pl.DataFrame()
        if self.opt("synthetic"):
            grid = me.scenario_grid()
            ck = ae.experiment_checkpoint(self.store, "m12b_synthetic", C, [], {"scenarios": [s.name for s in grid], "reps": C.sim_reps})
            prog = m11c.Progress(len(grid), "12B synthetic", every_s=20)
            for s in grid:
                if not ck.has(s.name):
                    ck.save(s.name, me.run_scenario(s, C, C.seed))
                prog.step(note=s.name)
            ae.require_complete(ck, [s.name for s in grid], "12B synthetic grid")
            SIM = ck.collect([s.name for s in grid])
            SIMS = me.scenario_summary(SIM)
        else:
            md("_Synthetic recovery grid not run (options['12B']['synthetic'] is off)._")
        self.SIM, self.SIMS = SIM, SIMS
        self.W(SIM, "m12b_synthetic_replicates")
        self.W(SIMS, "m12b_synthetic_summary")
        if SIMS.height:
            display_df(SIMS, rows=60)
            self.add_report_table("Synthetic recovery (estimator behaviour only)", SIMS)

    @step
    def bayes(self) -> None:
        """Part 8a (B4 optional) — PyMC latent model; reported only with R-hat ≤ 1.01 and bulk ESS ≥ 400."""
        import polars as pl
        from nflbdb.research.advanced_movement.m12 import advanced_evaluation as ae
        from nflbdb.research.advanced_movement.m12 import measurement_error as me

        self.BAYES = {}
        if self.opt("bayes") and me.bayes_available():
            off, stats, long, nw = self.off, self.stats, self.long, self.nw
            p = me.PREREGISTERED[4]
            d = off.join(stats.filter(pl.col("feature") == p.feature), on="nfl_id").join(nw.select("nfl_id", p.target), on="nfl_id").filter(pl.col(p.target).is_not_null())
            xs = [g["value"].to_numpy() for _, g in sorted(long.filter((pl.col("feature") == p.feature) & pl.col("nfl_id").is_in(d["nfl_id"].to_list())).partition_by(["nfl_id"], as_dict=True).items())]
            d = d.sort("nfl_id")
            self.BAYES = me.fit_bayes(xs, d[p.target].to_numpy(), me._covariates(d, list(ae.M1_COLS)), seed=self.C.seed)
            if self.controls.save_tables:
                self.store.write_json("m12b_bayes", {"feature": p.feature, "target": p.target, **self.BAYES})
            display_df(pl.DataFrame(self.BAYES["summary"]))
            print(f"max R-hat {self.BAYES['r_hat_max']:.3f} · min bulk ESS {self.BAYES['ess_bulk_min']:.0f} (R-hat > 1.01 or ESS < 400 → do not interpret)")
        else:
            print("B4 Bayesian model:", "not requested (options['12B']['bayes'] is off)" if not self.opt("bayes") else "PyMC not installed — not_feasible")

    @step
    def figures(self) -> None:
        """Parts 8b–9 — naive vs corrected coefficients, synthetic bias / coverage, cross-conformal calibration, posterior shrinkage."""
        import numpy as np
        import polars as pl
        from nflbdb.research.advanced_movement.m12 import figures as fg
        from nflbdb.research.advanced_movement.m12 import measurement_error as me

        store, EST, SIMS, OOF = self.store, self.EST, self.SIMS, self.OOF
        self.ok = ok = EST.filter(pl.col("status") == "computed") if "status" in EST.columns else EST
        cl = []
        for r in ok.iter_rows(named=True):
            for k in ("beta_obs", "beta_rc", "beta_simex", "beta_em"):
                if r.get(k) is not None:
                    cl.append({"label": f"{r['feature'].split('.', 1)[-1]} → {r['target']} · {k}", "estimator": k, "est": r[k], "lo": r.get(f"{k}_lo"), "hi": r.get(f"{k}_hi")})
        cl = pl.DataFrame(cl, infer_schema_length=None)
        store.show(fg.fig_forest(cl, "label", "est", "lo", "hi", "Naive versus corrected coefficients",
                                 "Standardized slope of the NFL target on the Combine feature given M1: naive (x̄), regression calibration, SIMEX, joint EM; athlete-bootstrap 95 % CI. A corrected coefficient is a latent-scale estimand, not a predictive gain.",
                                 "standardized slope", color_by="estimator"), "AM12_B_05_naive_vs_corrected", {"coefficients": cl}, publication=True, source="m12b_estimands")
        if SIMS.height:
            store.show(fg.fig_heatmap(SIMS, "scenario", "estimator", "bias", "Synthetic recovery: bias", "Mean estimate − true standardized β per scenario and estimator (planted data; software behaviour only).", fmt="{:.2f}", cmap="RdBu_r"),
                       "AM12_B_06_synthetic_bias", {"synthetic": SIMS}, source="m12b_synthetic_summary")
            if "coverage" in SIMS.columns:
                store.show(fg.fig_heatmap(SIMS.filter(pl.col("coverage").is_not_null()), "scenario", "estimator", "coverage", "Synthetic recovery: 95 % interval coverage",
                                          "Share of replicates whose interval contains the true β (nominal 0.95).", fmt="{:.2f}"), "AM12_B_07_synthetic_coverage", {}, source="m12b_synthetic_summary")
        cal = []
        if OOF.height:
            for (f_, t_), g in OOF.partition_by(["feature", "target"], as_dict=True).items():
                for m_ in ("pred_M1", "pred_B0", "pred_B1"):
                    if m_ not in g.columns:
                        continue
                    y, p_, fo = g["y"].to_numpy(), g[m_].to_numpy(), g["fold"].to_numpy()
                    for lvl in (0.5, 0.8, 0.9):
                        cov = []
                        for fk in np.unique(fo):
                            q = np.quantile(np.abs(y[fo != fk] - p_[fo != fk]), lvl)
                            cov += (np.abs(y[fo == fk] - p_[fo == fk]) <= q).tolist()
                        cal.append({"feature": f_, "target": t_, "model": m_.removeprefix("pred_"), "nominal": lvl, "coverage": float(np.mean(cov))})
        cal = pl.DataFrame(cal)
        self.W(cal, "m12b_interval_calibration")
        if cal.height:
            cg = cal.group_by("model", "nominal").agg(pl.col("coverage").mean()).with_columns(pl.col("nominal").alias("tau"), pl.col("model").alias("estimator"))
            store.show(fg.fig_calibration(cg, "Predictive interval calibration (cross-conformal)", "Held-out coverage of absolute-residual intervals built from the OTHER folds, averaged over feature × target pairs, for M1, B0 and B1."),
                       "AM12_B_08_interval_calibration", {"calibration": cg}, source="m12b_heldout_oof")
        p = me.PREREGISTERED[4]
        lf = self.long.filter(pl.col("feature") == p.feature)
        comp = me.components(lf["value"].to_numpy(), lf["nfl_id"].to_numpy())
        if comp["identifiable"]:
            post = self.stats.filter(pl.col("feature") == p.feature).with_columns((comp["sigma2_theta"] / (comp["sigma2_theta"] + comp["sigma2_eps"] / pl.col("n_att"))).alias("lam"))
            post = post.with_columns((comp["mean"] + pl.col("lam") * (pl.col("xbar") - comp["mean"])).alias("posterior_mean"),
                                     (pl.col("lam") * comp["sigma2_eps"] / pl.col("n_att")).sqrt().alias("posterior_sd")).join(self.off.select("nfl_id", "position_group"), on="nfl_id")
            store.show(fg.fig_scatter(post, "xbar", "posterior_mean", "Athlete posterior uncertainty (regression calibration)",
                                      f"Observed athlete mean vs E[θ | x̄] for {p.feature} (pooled components); shrinkage toward the mean grows as attempts fall. Posterior SD in the table.",
                                      "observed athlete mean", "posterior mean of the latent value", ref_line="identity"), "AM12_B_09_posterior_uncertainty", {"posterior": post}, source="m12b_reliability")

    @step
    def conclude(self) -> None:
        """Parts 10–12 — comparisons + evidence for 12G, conclusion, documents, run manifest."""
        import polars as pl
        from nflbdb.research.advanced_movement.m12 import advanced_evaluation as ae
        from nflbdb.research.advanced_movement.m12 import config as m12c
        from nflbdb.research.advanced_movement.m12 import reporting as rp
        from nflbdb.research.advanced_movement.m12 import synthesis as sy

        C, rm, HELD, ok, rt, ra, long, stats = self.C, self.controls.run_mode, self.HELD, self.ok, self.rt, self.ra, self.long, self.stats
        comps, ev_rows = [], []
        for r in (HELD.iter_rows(named=True) if HELD.height else []):
            comps.append(sy.comparison_row("12B", f"{r['model']} vs {r['reference']}: {r['feature']} → {r['target']}", "Δ RMSE", r["delta_rmse"], r["delta_lo"], r["delta_hi"], r["n_athletes"],
                                           m12c.PHYSICAL_TRANSFER, "m12b_heldout", rm, experiment=2, primary=(r["model"], r["reference"]) == ("B1", "B0")))
        for r in ok.iter_rows(named=True):
            better = "positive" if r.get("expected_sign", 1) > 0 else "negative"
            for k in ("beta_obs", "beta_rc", "beta_simex", "beta_em"):
                if r.get(k) is not None:
                    comps.append(sy.comparison_row("12B", f"{k}: {r['feature']} → {r['target']}", k, r[k], r.get(f"{k}_lo"), r.get(f"{k}_hi"), r["n_athletes"], m12c.PHYSICAL_TRANSFER,
                                                   "m12b_estimands", rm, experiment=2, better=better))
            h = HELD.filter((pl.col("feature") == r["feature"]) & (pl.col("target") == r["target"]) & (pl.col("model") == "B1") & (pl.col("reference") == "M1")) if HELD.height else HELD
            h = h.row(0, named=True) if h.height else {}
            rr = rt.filter((pl.col("feature") == r["feature"]) & (pl.col("group") == "ALL"))
            rr = rr.row(0, named=True) if rr.height else {}
            ev_rows.append(ae.evidence_row(f"12B.RC.{r['feature']}→{r['target']}", "12B", experiment="2", physical_interpretation="regression-calibrated athlete value of the feature",
                                           input_data="m11_attempt_features", output=r["target"], n_athletes=r["n_athletes"], reliability=rr.get("icc"), reliability_lo=rr.get("icc_lo"),
                                           reliability_hi=rr.get("icc_hi"), cross_domain_stat=r.get("r_partial"), cross_domain_lo=r.get("r_partial_lo"), cross_domain_hi=r.get("r_partial_hi"),
                                           delta_rmse=h.get("delta_rmse"), delta_lo=h.get("delta_lo"), delta_hi=h.get("delta_hi"), compute_cost="seconds",
                                           evaluation_target=m12c.PHYSICAL_TRANSFER, training_regime=m12c.INDUCTIVE, run_mode=rm, config_hash=C.hash()))
        COMP, EVID = sy.comparison_frame(comps), ae.evidence_frame(ev_rows)
        self.W(COMP, "m12b_comparisons")
        self.W(EVID, "m12b_evidence")
        prim = [r["verdict"] for r in COMP.filter(pl.col("primary")).iter_rows(named=True)]
        status = rp.status_from(prim)
        facts = {"status": status, "run_mode": rm,
                 "sample": f"{long['nfl_id'].n_unique()} athletes with preregistered attempt values; {stats.filter(pl.col('n_att') >= 2)['nfl_id'].n_unique()} with a repeat attempt of ≥ 1 feature.",
                 "evidence": [f"{r['feature']}: ICC {rp.ci(r['icc'], r['icc_lo'], r['icc_hi'])} (n={r['n_athletes']}, repeats {r['n_repeat']})" for r in ra.iter_rows(named=True)]
                             + [f"{r['feature']} → {r['target']}: β_obs {rp.ci(r.get('beta_obs'), r.get('beta_obs_lo'), r.get('beta_obs_hi'))}, β_RC {rp.ci(r.get('beta_rc'), r.get('beta_rc_lo'), r.get('beta_rc_hi'))}, "
                                f"compatible latent r [{rp.fmt(r.get('latent_lo'))}, {rp.fmt(r.get('latent_hi'))}], detectable latent r ≥ {rp.fmt(r.get('mde_latent'))}" for r in ok.iter_rows(named=True)]
                             + [f"held-out B1 vs B0 ({r['feature']} → {r['target']}): Δ RMSE {rp.ci(r['delta_rmse'], r['delta_lo'], r['delta_hi'], 4)} — {r['verdict']}"
                                for r in (HELD.filter(pl.col("model") == "B1").filter(pl.col("reference") == "B0").iter_rows(named=True) if HELD.height else [])],
                 "alternatives": ["Errors may be differential (a nervous athlete is slow in every attempt and in games): RC / SIMEX then over-correct (see the synthetic correlated-error scenario).",
                                  "Repeat attempts are not missing at random (fatigue, injury, invitation): σ²_ε may be underestimated (see the MNAR scenario)."],
                 "relevance": "Distinguishes “the Combine feature is too noisy to show a relation” from “there is no relation of a detectable size”.",
                 "limitations": ["1–3 attempts per athlete: within-athlete variance is imprecise.", "Corrections assume classical nondifferential error.", "One cohort."]
                                + ([] if self.opt("synthetic") else ["Synthetic recovery grid not run in this run mode."]),
                 "next": ["If compatible latent ranges are narrow and near 0 → stop correction work; otherwise collect more repeats per athlete."]}
        text = rp.conclusion_markdown("12B", facts)
        md(text)
        rp.write_conclusion(self.store, "12B", text)
        nl = chr(10)
        rp.update_doc(self.store, "MEASUREMENT_ERROR.md", text + nl + "### Estimands" + nl + nl + rp.md_table(ok, ["feature", "target", "n_athletes", "beta_obs", "beta_rc", "beta_simex", "beta_em", "latent_lo", "latent_hi", "mde_latent"]))
        rp.run_manifest(self.store, "12B", self.ctx.provenance("12B"), ["m12b_reliability", "m12b_estimands", "m12b_heldout", "m12b_synthetic_summary", "m12b_comparisons", "m12b_evidence"], status)
        self.facts, self.conclusion_md, self.m12_status = facts, text, status

    def run_all(self) -> None:
        for s in (self.prepare, self.reliability, self.corrections, self.synthetic, self.bayes, self.figures, self.conclude):
            s()


__all__ = ["MeasurementError"]
