"""12D — multi-view latent athletic traits (ported from ``12d_multiview_latent_athletic_traits.ipynb``).

Do several Combine drills, treated as measurement views, jointly reveal reproducible physical traits that single drill
metrics measure unreliably? Primary (reliability): cross-fitted shared factor scores have higher attempt-split
test–retest reliability than the most reliable single feature loading on them. Secondary (transfer): factor scores add
held-out information beyond official tests (D0 = M1) and beyond the concatenated features (D1).
"""

from __future__ import annotations

from .base import Stage, display_df, md, step


class MultiviewTraits(Stage):
    track = "12D"
    title = "Multi-view latent athletic traits"
    report_name = "12d_multiview_traits"
    methodology = (
        "Views = tracked drill families (`multiview_traits.VIEW_FAMILIES`) with ≤ 4 physics-preselected features each; official stopwatch tests "
        "are D0, never a view. Athletes of the Combine registry with ≥ 2 tracked families (`elig_12D`); folds = 12.0 athlete folds (athletes "
        "without one get a deterministic id-based fold for the unsupervised cross-fitting and never enter an NFL comparison). Models: D2 "
        "probabilistic FA · D3 structured multi-view FA (shared + view-specific factors) · D4 = D3 on position-centred features · D5 nonlinear "
        "only if its declared support rule passes. Features standardized with TRAINING-fold constants inside each fit; K by held-out "
        "(athlete-fold) log-likelihood per observed value (no NFL value). Missing views are not missing at random: scores use observed views only, "
        "missingness is diagnosed (SMD of officials within position), complete cases are a sensitivity. PRIMARY: cross-fitted attempt-split "
        "test–retest of factor scores minus that of the best single feature (paired athlete bootstrap). Stability: bootstrap Procrustes-aligned "
        "Tucker congruence; position loadings. Selection rule (declared, no NFL value) → `m12d_selected_representation.json`. Secondary transfer "
        "ladder D0 / D1 / D2–D4 on identical athletes per target.")

    @step
    def prepare(self) -> None:
        """Part 3 — data, views, availability and missingness."""
        import numpy as np
        import polars as pl
        from nflbdb.research.advanced_movement.m12 import advanced_evaluation as ae
        from nflbdb.research.advanced_movement.m12 import foundation as fd
        from nflbdb.research.advanced_movement.m12 import multiview_traits as mv

        C, D = self.C, self.D
        fd.check_foundation(self.store, C)
        self.reg = D.m12("m12_cohort_registry")
        self.off, athf, att = D.officials(), D.athlete_features(), D.attempt_features()
        self.POS = POS = {int(i): p for i, p in zip(self.off["nfl_id"].to_list(), self.off["position_group"].to_list())}
        self.views = views = mv.view_features(athf["feature"].unique().to_list())
        elig = self.reg.filter(pl.col("elig_12D"))["nfl_id"].to_list()
        self.ids, self.X, self.feats, self.views_of = mv.athlete_matrix(athf, views, elig)
        self.pos = np.array([POS.get(int(i), "?") for i in self.ids])
        fmap = dict(zip(*D.m12("m12_fold_manifest").select("nfl_id", "fold").to_dict(as_series=False).values()))
        self.folds = np.array([fmap.get(int(i), int(i) % C.outer_folds) if fmap.get(int(i)) is not None else int(i) % C.outer_folds for i in self.ids])
        self.X1, self.X2 = mv.attempt_split_matrices(att, self.ids, self.feats)
        display_df(ae.memory_plan([{"item": "athlete × feature matrix", "n_rows": self.X.shape[0], "n_cols": self.X.shape[1]}], self.knob("memory_budget_gb")))
        vt = pl.DataFrame([{"view": v, "features": ", ".join(fs)} for v, fs in views.items()])
        display_df(vt)
        self.avail = mv.availability(self.ids, self.X, self.views_of, POS)
        self.miss = mv.missingness_diagnostics(self.avail, self.off, list(views))
        self.vae = mv.vae_feasible(self.avail, C)
        self.W(self.avail, "m12d_availability")
        self.W(self.miss, "m12d_missingness")
        print(f"{self.ids.size} athletes · {len(self.feats)} features in {len(views)} views · D5:", self.vae["reason"])
        display_df(self.miss)
        self.add_report_table("Views and features", vt)

    @step
    def dimensionality(self) -> None:
        """Parts 4–6 — K per model by held-out log-likelihood (nothing NFL-related enters)."""
        import polars as pl
        from nflbdb.research.advanced_movement.m12 import multiview_traits as mv

        C, X, views_of, pos, folds = self.C, self.X, self.views_of, self.pos, self.folds
        self.ll = pl.concat([mv.heldout_loglik("D2", range(1, C.fa_max_k + 1), X, views_of, pos, folds, C, C.seed),
                             mv.heldout_loglik("D3", range(1, min(3, C.fa_max_k) + 1), X, views_of, pos, folds, C, C.seed),
                             mv.heldout_loglik("D4", range(1, min(3, C.fa_max_k) + 1), X, views_of, pos, folds, C, C.seed)])
        self.W(self.ll, "m12d_loglik")
        display_df(self.ll)
        self.BEST_K = {m: int(g.sort("heldout_loglik_per_value", descending=True)["k"][0]) for (m,), g in self.ll.partition_by(["model"], as_dict=True).items()}
        print("K by held-out likelihood:", self.BEST_K)
        self.add_report_table("Held-out log-likelihood per model and K", self.ll)

    @step
    def reliability(self) -> None:
        """Part 7a (PRIMARY) — full-data fits (display), variance shares, cross-fitted test–retest, bootstrap stability, position loadings,
        complete-case sensitivity, selected representation."""
        import numpy as np
        import polars as pl
        from nflbdb.research.advanced_movement.m12 import multiview_traits as mv

        C, X, views_of, pos, ids, feats, folds = self.C, self.X, self.views_of, self.pos, self.ids, self.feats, self.folds
        fits, retests, stab, posl = {}, [], [], []
        ft = None
        for m, k in self.BEST_K.items():
            fits[m], Xin = mv.fit_model(m, k, X, views_of, pos, np.ones(ids.size, bool), C, C.seed)
            rt, ft = mv.latent_test_retest(m, k, ids, X, self.X1, self.X2, views_of, feats, pos, folds, C, C.seed)
            retests.append(rt)
            if self.opt("bootstrap"):
                stab.append(mv.bootstrap_stability(m, k, X, views_of, pos, C, C.seed))
            posl.append(mv.position_loadings(m, k, X, views_of, pos, C, C.seed).with_columns(pl.lit(m).alias("model")))
        self.fits = fits
        self.RET = pl.concat(retests, how="diagonal_relaxed")
        STAB = pl.concat(stab) if stab else pl.DataFrame()
        POSL = pl.concat([p for p in posl if p.height], how="diagonal_relaxed") if any(p.height for p in posl) else pl.DataFrame()
        self.W(self.RET, "m12d_retest")
        self.W(ft, "m12d_feature_retest")
        self.W(STAB, "m12d_bootstrap_stability")
        self.W(POSL, "m12d_position_loadings")
        self.load = pl.concat([pl.DataFrame({"model": m, "feature": feats, "view": views_of, **{n: f.W[:, j] for j, n in enumerate(f.names)}, "psi": f.psi}) for m, f in fits.items()], how="diagonal_relaxed")
        self.vshare = pl.concat([pl.DataFrame({"model": m, "feature": feats, "shared": f.shared_variance_share()[:, 0], "view_specific": f.shared_variance_share()[:, 1],
                                               "unique": f.shared_variance_share()[:, 2]}) for m, f in fits.items()])
        self.W(self.load, "m12d_loadings")
        self.W(self.vshare, "m12d_variance_share")
        # complete-case sensitivity: athletes with every view that ≥ 80 % of their position performs
        typ = {p: [v for v in self.views if v in g.columns and g[v].mean() >= 0.8] for (p,), g in self.avail.partition_by(["position_group"], as_dict=True).items()}
        cc = np.array([all(r[v] for v in typ.get(r["position_group"], [])) for r in self.avail.iter_rows(named=True)])
        cc_rows = []
        for m, k in self.BEST_K.items():
            if cc.sum() >= 50:
                fcc, _ = mv.fit_model(m, k, X[cc], views_of, pos[cc], np.ones(int(cc.sum()), bool), C, C.seed)
                ks = [j for j, n in enumerate(fits[m].names) if n.startswith("shared")]
                B_ = mv.procrustes(fits[m].W[:, ks], fcc.W[:, ks]) if len(ks) > 1 else fcc.W[:, ks]
                cc_rows += [{"model": m, "factor": f"shared{q}", "n_complete": int(cc.sum()), "tucker_vs_all": mv.tucker(fits[m].W[:, ks][:, q], B_[:, q])} for q in range(len(ks))]
        CC = pl.DataFrame(cc_rows)
        self.W(CC, "m12d_complete_case")
        self.sel = mv.selection_rule(self.RET, self.ll)
        if self.controls.save_tables and self.sel.get("model"):
            self.store.write_json("m12d_selected_representation", {**self.sel, "rule": mv.selection_rule.__doc__.strip(), "config_hash": C.hash(), "features": feats, "views": views_of})
        display_df(self.RET)
        display_df(STAB.group_by("model", "factor").agg(pl.col("tucker").median().alias("median_tucker"), pl.col("tucker").quantile(0.05).alias("q05_tucker")) if STAB.height else STAB)
        display_df(POSL)
        display_df(CC)
        print("selected representation (declared rule, no NFL value):", self.sel)
        self.add_report_table("Cross-fitted test–retest: latent trait vs best single feature (PRIMARY)", self.RET,
                              ["model", "factor", "n_athletes", "retest_factor", "best_feature", "retest_best_feature", "gain", "gain_lo", "gain_hi"])

    @step
    def transfer(self) -> None:
        """Part 7b (secondary) — D0 / D1 / D2–D4 transfer ladder, identical athletes per target."""
        import polars as pl
        from nflbdb.research.advanced_movement.m11 import speed_state as ss
        from nflbdb.research.advanced_movement.m12 import advanced_evaluation as ae
        from nflbdb.research.advanced_movement.m12 import foundation as fd
        from nflbdb.research.advanced_movement.m12 import multiview_traits as mv

        C, D, store, ids, X, feats, views_of, POS = self.C, self.D, self.store, self.ids, self.X, self.feats, self.views_of, self.POS
        nw = ss.nfl_wide(D.nfl_targets())
        envw = ss.envelope_wide(D.envelopes(), "lin").select("nfl_id", "env.lin_A0")
        tgts = nw.join(envw, on="nfl_id", how="full", coalesce=True)
        if store.has("m12c_targets"):
            tgts = tgts.join(store.read_frame("m12c_targets", fd.NB["12C"]).select("nfl_id", "nfl.q90_shared"), on="nfl_id", how="full", coalesce=True)
        fman = D.m12("m12_fold_manifest").select("nfl_id", "fold")
        featw = pl.DataFrame({"nfl_id": ids, **{f: X[:, j] for j, f in enumerate(feats)}})
        linked = self.reg.filter(pl.col("linked"))["nfl_id"].to_list()
        self.data = self.off.filter(pl.col("nfl_id").is_in(linked)).join(featw, on="nfl_id", how="inner").join(tgts, on="nfl_id", how="left").join(fman, on="nfl_id", how="inner").sort("nfl_id")
        MODELS = {"D0": ae.Model(list(ae.M1_COLS)), "D1": ae.Model([*ae.M1_COLS, *feats])}
        for m, k in self.BEST_K.items():
            MODELS[m] = ae.Model(list(ae.M1_COLS), factory=lambda a, _m=m, _k=k: mv.factor_block(_m, _k, ids, X, views_of, POS, a, C, C.seed))
        TR = []
        for tg in [t for t in ("nfl.q90_shared", "nfl.band_4_6", "env.lin_A0", "nfl.brake_decel") if t in self.data.columns]:
            res = ae.oof_ladder(self.data, tg, MODELS, C, C.seed)
            TR.append(ae.compare(res, [(m, "D0") for m in MODELS if m != "D0"] + [(m, "D1") for m in self.BEST_K], C, C.seed, target=tg))
        self.TR = pl.concat([t for t in TR if t.height], how="diagonal_relaxed") if TR else pl.DataFrame()
        self.W(self.TR, "m12d_transfer")
        t = self.TR.select("target", "model", "reference", "n_athletes", "delta_rmse", "delta_lo", "delta_hi", "verdict") if self.TR.height else self.TR
        display_df(t)
        self.add_report_table("NFL transfer of multi-view traits (secondary)", t)

    @step
    def figures(self) -> None:
        """Parts 8–9 — availability, loadings, variance share, position profiles, repeatability forest, latent space, transfer,
        trait uncertainty, missingness."""
        import numpy as np
        import polars as pl
        from nflbdb.research.advanced_movement.m12 import figures as fg
        from nflbdb.research.advanced_movement.m12 import multiview_traits as mv

        store, avail, views, ids, pos, X, feats = self.store, self.avail, self.views, self.ids, self.pos, self.X, self.feats
        av_long = avail.unpivot(index=["nfl_id", "position_group"], on=list(views), variable_name="view", value_name="observed").group_by("position_group", "view").agg(pl.col("observed").mean().alias("participation"))
        store.show(fg.fig_heatmap(av_long, "position_group", "view", "participation", "Multi-view availability", "Share of athletes per position with a tracked attempt in each view (participation is not random).", fmt="{:.2f}"),
                   "AM12_D_01_view_availability", {"availability": av_long}, publication=True, source="m12d_availability")
        msel = self.sel.get("model") or next(iter(self.fits))
        f_ = self.fits[msel]
        store.show(fg.fig_matrix(f_.W, feats, f_.names, f"Factor loading matrix ({msel}, K = {self.BEST_K[msel]})",
                                 "Standardized loadings of the full-data fit (display; analyses are cross-fitted). Sign convention: largest |loading| positive per factor; factors are not named without support.", "loading"),
                   "AM12_D_02_loading_matrix", {"loadings": self.load}, publication=True, source="m12d_loadings")
        store.show(fg.fig_stacked(self.vshare.filter(pl.col("model") == msel), "feature", ["shared", "view_specific", "unique"], "Shared versus unique variance",
                                  f"Per feature, share of standardized variance explained by shared factors, the view-specific factor and unique variance ({msel}).", "share of variance"),
                   "AM12_D_03_variance_share", {"variance": self.vshare}, source="m12d_variance_share")
        Sc = f_.scores(mv.position_center(X, pos, np.ones(ids.size, bool)) if msel == "D4" else X)
        ks = [j for j, n in enumerate(f_.names) if n.startswith("shared")]
        prof = []
        for p in np.unique(pos):
            mm = (pos == p) & np.isfinite(Sc[:, ks[0]])
            if mm.sum() >= 15:
                for q in ks:
                    v = Sc[mm, q]
                    se = v.std() / np.sqrt(v.size)
                    prof.append({"label": f"{p} · {f_.names[q]} (n={int(mm.sum())})", "position_group": str(p), "mean": float(v.mean()), "lo": float(v.mean() - 1.96 * se), "hi": float(v.mean() + 1.96 * se)})
        prof = pl.DataFrame(prof)
        store.show(fg.fig_forest(prof, "label", "mean", "lo", "hi", "Position-specific latent profiles",
                                 "Mean shared factor score by position with 95 % CI (standardized units; positions with ≥ 15 athletes). Profiles are conditional on drill participation.", "factor score",
                                 color_by="position_group"), "AM12_D_04_position_profiles", {"profiles": prof}, source="m12d (full-data fit)")
        rr = self.RET.with_columns((pl.col("model") + " · " + pl.col("factor") + " vs " + pl.col("best_feature").fill_null("—")).alias("label"))
        store.show(fg.fig_forest(rr, "label", "gain", "gain_lo", "gain_hi", "Attempt-level repeatability: latent trait versus best single feature",
                                 "Test–retest r of cross-fitted factor scores (first vs second attempts) minus that of the best single feature loading on the factor, same athletes, athlete bootstrap 95 % CI (PRIMARY).",
                                 "Δ test–retest r", color_by="model"), "AM12_D_05_latent_vs_feature_repeatability", {"retest": rr}, publication=True, source="m12d_retest")
        if len(ks) >= 2:
            sp = pl.DataFrame({"f0": Sc[:, ks[0]], "f1": Sc[:, ks[1]], "position_group": pos})
            store.show(fg.fig_scatter(sp, "f0", "f1", "Latent-space structure", "First two shared factor scores (full-data fit) coloured by position.", f_.names[ks[0]], f_.names[ks[1]]),
                       "AM12_D_06_latent_space", {}, source="m12d")
        if self.TR.height:
            tf = self.TR.with_columns((pl.col("target") + " · " + pl.col("model") + " vs " + pl.col("reference") + " (n=" + pl.col("n_athletes").cast(pl.String) + ")").alias("label"))
            store.show(fg.fig_forest(tf, "label", "delta_rmse", "delta_lo", "delta_hi", "NFL transfer of multi-view traits", "Δ RMSE (reference − model), identical held-out athletes per target, paired athlete bootstrap.",
                                     "Δ RMSE", color_by="model"), "AM12_D_07_transfer", {"transfer": tf}, publication=True, source="m12d_transfer")
        unc = []
        Xin_ = mv.position_center(X, pos, np.ones(ids.size, bool)) if msel == "D4" else X
        Z = (Xin_ - f_.center) / f_.scale
        for i in range(ids.size):
            o = np.isfinite(Z[i])
            if o.any():
                Minv = np.linalg.inv(np.eye(f_.W.shape[1]) + (f_.W[o].T / f_.psi[o]) @ f_.W[o])
                unc.append({"nfl_id": int(ids[i]), "position_group": str(pos[i]), "n_features_observed": int(o.sum()), "posterior_sd_f0": float(np.sqrt(Minv[ks[0], ks[0]]))})
        unc = pl.DataFrame(unc)
        self.W(unc, "m12d_trait_uncertainty")
        store.show(fg.fig_scatter(unc, "n_features_observed", "posterior_sd_f0", "Interpretable trait uncertainty",
                                  "Posterior SD of the first shared factor score against the number of observed features: athletes with fewer drills are less certain.", "observed features", "posterior SD (factor units)"),
                   "AM12_D_08_trait_uncertainty", {"uncertainty": unc}, source="m12d")
        sm_ = self.miss.unpivot(index=["position_group", "view"], on=["smd_forty", "smd_weight", "smd_height"], variable_name="official", value_name="smd").with_columns((pl.col("position_group") + " · " + pl.col("official")).alias("row"))
        store.show(fg.fig_heatmap(sm_, "row", "view", "smd", "Missingness sensitivity: who performs each drill",
                                  "Standardized mean difference (participants − non-participants) of official forty, weight, height within position. |SMD| ≥ 0.2 = participation related to measured athleticism.",
                                  fmt="{:.2f}", cmap="RdBu_r"), "AM12_D_09_missingness", {"missingness": sm_}, source="m12d_missingness")

    @step
    def conclude(self) -> None:
        """Parts 10–12 — comparisons + evidence for 12G, conclusion, documents, run manifest."""
        import polars as pl
        from nflbdb.research.advanced_movement.m12 import advanced_evaluation as ae
        from nflbdb.research.advanced_movement.m12 import config as m12c
        from nflbdb.research.advanced_movement.m12 import reporting as rp
        from nflbdb.research.advanced_movement.m12 import synthesis as sy

        C, rm, RET, TR, sel = self.C, self.controls.run_mode, self.RET, self.TR, self.sel
        comps = [sy.comparison_row("12D", f"retest {r['model']} {r['factor']} vs best feature {r['best_feature']}", "Δ test–retest r", r["gain"], r["gain_lo"], r["gain_hi"], r["n_athletes"],
                                   m12c.RELIABILITY, "m12d_retest", rm, experiment=4, primary=r["model"] == sel.get("model")) for r in RET.iter_rows(named=True)]
        comps += [sy.comparison_row("12D", f"transfer {r['model']} vs {r['reference']} → {r['target']}", "Δ RMSE", r["delta_rmse"], r["delta_lo"], r["delta_hi"], r["n_athletes"],
                                    m12c.PHYSICAL_TRANSFER, "m12d_transfer", rm, experiment=4) for r in (TR.iter_rows(named=True) if TR.height else [])]
        COMP = sy.comparison_frame(comps)
        self.W(COMP, "m12d_comparisons")
        ev_rows = []
        for r in RET.iter_rows(named=True):
            t = TR.filter((pl.col("model") == r["model"]) & (pl.col("reference") == "D0")) if TR.height else TR
            t = t.sort("n_athletes", descending=True).row(0, named=True) if t.height else {}
            ev_rows.append(ae.evidence_row(f"12D.{r['model']}.{r['factor']}", "12D", experiment="4", physical_interpretation="shared multi-drill factor (unnamed unless supported)", input_data="m11_athlete_features (views)",
                                           output="factor score", n_athletes=r["n_athletes"], reliability=r["retest_factor"], delta_rmse=t.get("delta_rmse"), delta_lo=t.get("delta_lo"), delta_hi=t.get("delta_hi"),
                                           compute_cost="tens of minutes", evaluation_target=m12c.RELIABILITY, training_regime=m12c.INDUCTIVE, run_mode=rm, config_hash=C.hash()))
        EVID = ae.evidence_frame(ev_rows)
        self.W(EVID, "m12d_evidence")
        status = rp.status_from(COMP.filter(pl.col("primary"))["verdict"].to_list())
        facts = {"status": status, "run_mode": rm, "sample": f"{self.ids.size} athletes, {len(self.feats)} features in {len(self.views)} views; {self.data.height} linked for transfer.",
                 "evidence": [f"K by held-out likelihood: {self.BEST_K}; selected {sel.get('model')} (K={sel.get('k')})", self.vae["reason"]]
                             + [f"{r['model']} {r['factor']}: retest {rp.fmt(r['retest_factor'])} vs {r['best_feature']} {rp.fmt(r['retest_best_feature'])}, Δ {rp.ci(r['gain'], r['gain_lo'], r['gain_hi'])} (n={r['n_athletes']})"
                                for r in RET.iter_rows(named=True)]
                             + [f"{r['model']} vs {r['reference']} → {r['target']}: Δ RMSE {rp.ci(r['delta_rmse'], r['delta_lo'], r['delta_hi'], 4)} — {r['verdict']}" for r in (TR.iter_rows(named=True) if TR.height else [])],
                 "alternatives": ["A higher factor reliability can come from averaging several noisy features (Spearman–Brown), not from a new trait.",
                                  "Shared factors may encode position / body size rather than movement mechanics (compare D3 and D4)."],
                 "relevance": "Tells whether several imperfect drills can be combined into one more stable, interpretable physical measurement.",
                 "limitations": ["Views are missing not at random.", "Few repeats per drill.", "Loadings rotate within the shared block; interpretation only with stability support."]
                                + ([] if self.opt("bootstrap") else ["Bootstrap loading stability not run in this run mode."]),
                 "next": ["If a factor is more reliable but not predictive, use it descriptively (scouting profile), not as an NFL forecast."]}
        text = rp.conclusion_markdown("12D", facts)
        md(text)
        rp.write_conclusion(self.store, "12D", text)
        nl = chr(10)
        rp.update_doc(self.store, "MULTIVIEW_TRAITS.md", text + nl + "### Repeatability (cross-fitted)" + nl + nl + rp.md_table(RET, ["model", "factor", "n_athletes", "retest_factor", "best_feature", "retest_best_feature", "gain", "gain_lo", "gain_hi"]))
        rp.run_manifest(self.store, "12D", self.ctx.provenance("12D"), ["m12d_loglik", "m12d_retest", "m12d_loadings", "m12d_transfer", "m12d_comparisons", "m12d_evidence"], status, {"selected": sel})
        self.facts, self.conclusion_md, self.m12_status = facts, text, status

    def run_all(self) -> None:
        for s in (self.prepare, self.dimensionality, self.reliability, self.transfer, self.figures, self.conclude):
            s()


__all__ = ["MultiviewTraits"]
