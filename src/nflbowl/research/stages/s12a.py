"""12A — functional movement transfer (ported from ``12a_functional_movement_transfer.ipynb``).

Does modelling the whole Combine acceleration–speed function a_C(v) retain transferable athlete-level information
that scalar band features lose — and where along the speed axis does transfer sit? Primary: A2 (M1 + fold-fitted
FPCA scores of a_C(v)) vs A0 (M1 + scalar band accelerations) on the held-out athlete-mean squared error of the NFL
τ-quantile function a_N(v) (the 12C frozen estimator), identical athletes.
"""

from __future__ import annotations

from .base import Stage, display_df, md, step


class FunctionalTransfer(Stage):
    track = "12A"
    title = "Functional movement transfer"
    report_name = "12a_functional_transfer"
    methodology = (
        "Cohort: linked athletes with ≥ 60 % of the shared bins supported in both domains (`elig_12A`). a_C(v): frame-weighted mean rising-phase "
        "acceleration per shared speed bin over the athlete's forty attempts (one interior gap ≤ 1 bin bridged and counted; no extrapolation). "
        "a_N(v): the 12C frozen estimator from all the athlete's games. Representations: A0 scalar bands · A1 P-spline coefficients (λ chosen by "
        "Combine repeat-attempt agreement — no NFL value) · A2 fold-fitted FPCA scores (K = smallest reaching `fpca_var` of TRAINING variance, ≤ "
        "`fpca_max_k`) · A4 hierarchical training-only shrinkage; controls: Combine curves shuffled within position, random scores. Function-on-"
        "function ridge (FPC mode) and pointwise mode; athlete-mean squared error over each athlete's own supported bins (equal athlete weight); "
        "paired athlete bootstrap. Pointwise effect along speed; position subgroups; hierarchical attempt-1 → attempt-2 prediction; FPC "
        "test–retest. Exploratory: landmark registration (4 / 6 / 8 yd/s crossings) and SRSF elastic alignment of forty speed–time curves "
        "(phase = warp distance, amplitude = aligned shape; timing is reported, never discarded), their ICC, stopwatch redundancy and the "
        "stopwatch-twin decomposition.")

    @step
    def prepare(self) -> None:
        """Parts 3–4 — cohort, Combine athlete / attempt matrices, NFL functions (12C freeze), aligned design frame."""
        import numpy as np
        import polars as pl
        from nflbdb.research.advanced_movement.m11 import combine_drills as cd
        from nflbdb.research.advanced_movement.m12 import advanced_evaluation as ae
        from nflbdb.research.advanced_movement.m12 import distributional_capacity as dc
        from nflbdb.research.advanced_movement.m12 import foundation as fd
        from nflbdb.research.advanced_movement.m12 import functional_movement as fm

        C, D, store = self.C, self.D, self.store
        fd.check_foundation(store, C)
        self.FREEZE = FREEZE = dc.read_freeze(store)                 # dependency gate: 12C must have frozen its target
        self.BINS = BINS = FREEZE["shared_bins"]
        self.VMID = ae.mids(ae.speed_edges(C))[BINS]
        md(f"**Run mode: `{self.controls.run_mode}`** · config `{C.hash()}` · NFL function = 12C {FREEZE['estimator']} at τ = {FREEZE['tau']} · {len(BINS)} shared bins")
        reg = D.m12("m12_cohort_registry")
        self.off, athf = D.officials(), D.athlete_features()
        self.POS = POS = {int(i): p for i, p in zip(self.off["nfl_id"].to_list(), self.off["position_group"].to_list())}
        ids = np.array(sorted(reg.filter(pl.col("elig_12A"))["nfl_id"].to_list()))
        curves = D.m12("m12_combine_curves")
        self.ids_c, self.Mc, bridged = fm.combine_athlete_matrix(curves, BINS, C, ids)
        self.att_meta, self.A = fm.combine_attempt_matrix(curves.filter(pl.col("nfl_id").is_in(ids.tolist())), BINS, C)
        h = D.hist_athletes(ids.tolist())
        cubes = dc.build_cubes(h, POS, C)
        del h
        Fn = dc.curves(cubes, None, [FREEZE["tau"]], C, FREEZE["lambda_c2"], C.env_boot, C.seed)[FREEZE["tau"]][FREEZE["estimator"]]
        Y = np.array([Fn[int(i)][BINS] if int(i) in Fn else np.full(len(BINS), np.nan) for i in ids])
        display_df(ae.memory_plan([{"item": "Combine athlete curves", "n_rows": self.Mc.shape[0], "n_cols": self.Mc.shape[1]}, {"item": "attempt curves", "n_rows": self.A.shape[0], "n_cols": self.A.shape[1]},
                                   {"item": "NFL functions", "n_rows": Y.shape[0], "n_cols": Y.shape[1]}], self.knob("memory_budget_gb")))
        fman = D.m12("m12_fold_manifest").select("nfl_id", "fold")
        cw = cd.wide(athf, ["forty.band_2_4_mean_accel", "forty.band_4_6_mean_accel", "forty.band_6_8_mean_accel"])
        # rows MUST align with Y (built in sorted ids order): joins do not guarantee order, so sort afterwards
        df = pl.DataFrame({"nfl_id": ids}).join(self.off, on="nfl_id", how="left").join(cw, on="nfl_id", how="left").join(fman, on="nfl_id", how="left").sort("nfl_id")
        assert (df["nfl_id"].to_numpy() == ids).all()
        keep = (df["fold"].is_not_null() & pl.Series(np.isfinite(Y).sum(1) >= max(3, len(BINS) // 2))).to_numpy()
        self.df, self.Y = df.filter(pl.Series(keep)), Y[keep]
        self.ids = self.df["nfl_id"].to_numpy()
        print(f"{self.ids.size} athletes · {self.A.shape[0]} attempts · bridged interior bins: {int(bridged.sum())} over {int((bridged > 0).sum())} athletes")

    @step
    def representations(self) -> None:
        """Parts 5–6 — spline λ by repeat agreement; A0 / A1 / A2 / A4 + controls in FPC and pointwise modes."""
        import numpy as np
        import polars as pl
        from nflbdb.research.advanced_movement.m12 import advanced_evaluation as ae
        from nflbdb.research.advanced_movement.m12 import functional_movement as fm

        C, ids_c, Mc, POS = self.C, self.ids_c, self.Mc, self.POS
        self.Bsp, lo_, hi_ = fm.spline_basis_for(self.BINS, C)
        self.LAM_S, lam_tab = fm.choose_lambda_repeat(self.att_meta, self.A, self.Bsp, set(self.ids.tolist()), C.spline_lambdas)
        self.W(lam_tab, "m12a_spline_lambda")
        display_df(lam_tab)
        rng = np.random.default_rng(C.seed)
        posa = np.array([POS.get(int(i), "?") for i in ids_c])
        perm = np.arange(ids_c.size)
        for p in np.unique(posa):
            ix = np.flatnonzero(posa == p)
            perm[ix] = rng.permutation(ix)
        Mshuf = Mc[perm]
        self.f_full = fm.FPCA.fit(Mc, C.fpca_var, C.fpca_max_k)
        KD = self.f_full.phi.shape[1] if self.f_full is not None else 2

        def rand_block(a_ids):
            R = np.random.default_rng(C.seed + 5).standard_normal((len(a_ids), KD))
            return lambda train: pl.DataFrame({f"r{k}": R[:, k] for k in range(KD)})

        BANDS = ["forty.band_2_4_mean_accel", "forty.band_4_6_mean_accel", "forty.band_6_8_mean_accel"]
        Bsp, LAM_S, att_meta, A = self.Bsp, self.LAM_S, self.att_meta, self.A
        self.MODELS = MODELS = {"M1": ae.Model(list(ae.M1_COLS)), "A0": ae.Model([*ae.M1_COLS, *BANDS]),
                                "A1": ae.Model(list(ae.M1_COLS), factory=lambda a: fm.spline_block(a, ids_c, Mc, Bsp, LAM_S)),
                                "A2": ae.Model(list(ae.M1_COLS), factory=lambda a: fm.fpca_block(a, ids_c, Mc, C)),
                                "A4": ae.Model(list(ae.M1_COLS), factory=lambda a: fm.hierarchical_block(a, att_meta, A, POS, C)),
                                "CTRL_shuffled": ae.Model(list(ae.M1_COLS), factory=lambda a: fm.fpca_block(a, ids_c, Mshuf, C)),
                                "CTRL_random": ae.Model(list(ae.M1_COLS), factory=rand_block)}
        self.res = fm.functional_oof(self.df, self.Y, MODELS, C, C.seed, mode="fpc")
        comps = [("A2", "A0"), ("A0", "M1"), ("A1", "M1"), ("A2", "M1"), ("A4", "M1"), ("A2", "A1"), ("CTRL_shuffled", "M1"), ("CTRL_random", "M1")]
        self.cmp = fm.compare_functional(self.res, comps, C, C.seed).with_columns(pl.lit("fof").alias("mode"))
        res_pw = fm.functional_oof(self.df, self.Y, {k: MODELS[k] for k in ("M1", "A0", "A2")}, C, C.seed, mode="pointwise")
        cmp_pw = fm.compare_functional(res_pw, [("A2", "A0"), ("A0", "M1"), ("A2", "M1")], C, C.seed).with_columns(pl.lit("pointwise").alias("mode"))
        self.CMP = pl.concat([self.cmp, cmp_pw], how="diagonal_relaxed")
        self.W(self.CMP, "m12a_compare")
        t = self.CMP.select("mode", "model", "reference", "n_athletes", "mse_ref", "mse_model", "delta_mse", "delta_lo", "delta_hi", "delta_pct", "spearman_level_model", "verdict")
        display_df(t)
        self.add_report_table("Scalar versus functional representations (identical athletes)", t)

    @step
    def localisation(self) -> None:
        """Part 7 — pointwise effects along speed, positions, hierarchical repeat prediction, FPC test–retest."""
        import polars as pl
        from nflbdb.research.advanced_movement.m12 import advanced_evaluation as ae
        from nflbdb.research.advanced_movement.m12 import functional_movement as fm

        C, res = self.C, self.res
        self.pw = pl.concat([fm.pointwise_effect(res, a_, b_, self.VMID, C, C.seed).with_columns(pl.lit(f"{a_} vs {b_}").alias("comparison"))
                             for a_, b_ in (("A2", "A0"), ("A2", "M1"), ("A0", "M1"))])
        self.W(self.pw, "m12a_pointwise")
        bypos = []
        for p in sorted(set(res.positions.tolist())):
            m = res.positions == p
            if m.sum() < C.min_athletes // 2:
                continue
            a_, b_ = fm.athlete_mse(res.Y[m], res.preds["A0"][m]), fm.athlete_mse(res.Y[m], res.preds["A2"][m])
            bypos.append({"position_group": p, **ae.paired_metric_diff(a_, b_, max(200, C.n_boot // 5), C.seed)})
        self.bypos = pl.DataFrame(bypos, infer_schema_length=None)
        self.W(self.bypos, "m12a_by_position")
        rpt = fm.repeat_prediction(self.att_meta, self.A, self.POS)
        self.W(rpt, "m12a_hierarchical_repeat")
        self.rp_sum = ae.paired_metric_diff(rpt["mse_raw"].to_numpy(), rpt["mse_pooled"].to_numpy(), max(200, C.n_boot // 5), C.seed) if rpt.height else {}
        _, comps_h = fm.hierarchical_curves(self.att_meta, self.A, self.POS)
        self.W(comps_h, "m12a_hierarchical_components")
        fpc_rel = []
        if self.f_full is not None:
            am = self.att_meta.with_row_index("_r")
            pairs = [(g.sort("attempt_order")["_r"][0], g.sort("attempt_order")["_r"][1]) for _, g in am.partition_by(["nfl_id"], as_dict=True).items() if g.height >= 2]
            S1, S2 = self.f_full.scores(self.A[[a for a, _ in pairs]]), self.f_full.scores(self.A[[b for _, b in pairs]])
            for k in range(S1.shape[1]):
                fpc_rel.append({"component": k + 1, "variance_share_cum": float(self.f_full.var_share[k]), **ae.split_reliability(S1[:, k], S2[:, k], max(200, C.n_boot // 5), C.seed)})
        self.fpc_rel = pl.DataFrame(fpc_rel, infer_schema_length=None)
        self.W(self.fpc_rel, "m12a_fpc_reliability")
        display_df(self.bypos)
        display_df(pl.DataFrame([self.rp_sum])) if self.rp_sum else md("_no athletes with repeats_")
        display_df(self.fpc_rel)
        self.add_report_table("FPC attempt test–retest", self.fpc_rel)

    @step
    def elastic(self) -> None:
        """Part 7b — elastic functional alignment of forty speed–time curves (exploratory)."""
        import numpy as np
        import polars as pl
        from nflbdb.research.advanced_movement.m11 import evaluation as ev
        from nflbdb.research.advanced_movement.m12 import functional_movement as fm

        C, D = self.C, self.D
        pa, el_icc, el_red, twin_dec = pl.DataFrame(), pl.DataFrame(), pl.DataFrame(), pl.DataFrame()
        self.V = self.Va = self.G = self.g = None
        if self.opt("elastic"):
            st = D.m12("m12_combine_speed_time")
            meta_v, V, g = fm.speed_time_matrix(st, self.knob("elastic_window_s"), C)
            order = meta_v.with_row_index("_r").with_columns(pl.col("seq_id").hash(C.seed).alias("_h")).sort("_h").head(self.knob("elastic_max_attempts"))["_r"].to_numpy()
            order = np.sort(order)
            meta_v, V = meta_v[order], V[order]
            L = fm.landmarks(V, g)
            Lref = np.nanmean(L, 0)
            Vl = fm.landmark_register(V, g, L, Lref)                     # noqa: F841 — landmark registration (kept for parity with 12A)
            Va, G, tmpl = fm.elastic_align(V, g)
            pa = pl.concat([meta_v, fm.phase_amplitude(V, Va, G, tmpl, g), pl.DataFrame({"t_to_4": L[:, 0], "t_to_6": L[:, 1], "t_to_8": L[:, 2]})], how="horizontal")
            long_pa = pa.unpivot(index=["seq_id", "nfl_id"], on=["phase_distance", "amplitude_distance", "warp_mid_shift_s", "t_to_4", "t_to_6", "t_to_8"], variable_name="feature", value_name="value").drop_nulls()
            el_icc = ev.repeatability_table(long_pa, seed=C.seed)
            am = pa.group_by("nfl_id").agg([pl.col(k).mean() for k in ("phase_distance", "amplitude_distance", "warp_mid_shift_s", "t_to_4", "t_to_6", "t_to_8")]).join(self.off, on="nfl_id")
            el_red = pl.DataFrame([{"feature": k, **ev.stopwatch_redundancy(am, k)} for k in ("phase_distance", "amplitude_distance", "warp_mid_shift_s", "t_to_4", "t_to_8")])
            try:
                tw = D.twins()
            except FileNotFoundError:
                tw = pl.DataFrame()
                print("stopwatch twins not available: twin decomposition skipped")
            ath = {int(i): np.flatnonzero(meta_v["nfl_id"].to_numpy() == i) for i in np.unique(meta_v["nfl_id"].to_numpy())}
            rows = []
            for r in tw.iter_rows(named=True):
                a_, b_ = ath.get(int(r["nfl_id_a"])), ath.get(int(r["nfl_id_b"]))
                if a_ is None or b_ is None or not a_.size or not b_.size:
                    continue
                Ga, Gb = G[a_].mean(0), G[b_].mean(0)
                Vaa, Vbb = Va[a_].mean(0), Va[b_].mean(0)
                dt_ = g[1] - g[0]
                ph_, amp_ = float(np.sqrt(np.sum((Ga - Gb) ** 2) * dt_)), float(np.sqrt(np.sum((Vaa - Vbb) ** 2) * dt_))
                raw = float(np.sqrt(np.sum((V[a_].mean(0) - V[b_].mean(0)) ** 2) * dt_))
                rows.append({"pair_id": r["pair_id"], "position_group": r["position_group"], "d_forty": r["d_forty"], "phase_diff_s": ph_, "amplitude_diff": amp_, "raw_diff": raw})
            twin_dec = pl.DataFrame(rows)
            self.V, self.Va, self.G, self.g = V, Va, G, g
        else:
            md("_Elastic alignment not run (options['12A']['elastic'] is off)._")
        self.pa, self.el_icc, self.el_red, self.twin_dec = pa, el_icc, el_red, twin_dec
        self.W(pa, "m12a_phase_amplitude")
        self.W(el_icc, "m12a_elastic_repeatability")
        self.W(el_red, "m12a_elastic_redundancy")
        self.W(twin_dec, "m12a_twin_decomposition")
        if el_icc.height:
            display_df(el_icc)
            display_df(el_red)
            display_df(twin_dec.describe()) if twin_dec.height else md("_no twin pair with aligned attempts_")
            self.add_report_table("Elastic phase / amplitude repeatability", el_icc)

    @step
    def figures(self) -> None:
        """Parts 8–9 — raw / smoothed functions, FPCA components, predicted vs observed, pointwise effect, functional bands, forest, positions,
        phase / amplitude, twins."""
        import numpy as np
        import polars as pl
        from nflbdb.research.advanced_movement.m12 import figures as fg
        from nflbdb.research.advanced_movement.m12 import functional_movement as fm

        C, store, VMID, POS, res = self.C, self.store, self.VMID, self.POS, self.res
        ids_c, att_meta, A, Mc = self.ids_c, self.att_meta, self.A, self.Mc
        ex = [int(ids_c[k]) for k in np.linspace(0, ids_c.size - 1, 6).astype(int)] if ids_c.size else []
        series = []
        for i in ex:
            rows_ = np.flatnonzero(att_meta["nfl_id"].to_numpy() == i)
            lines = {f"attempt {k + 1}": A[r] for k, r in enumerate(rows_[:3])}
            k_ = int(np.flatnonzero(ids_c == i)[0])
            lines["athlete mean"] = Mc[k_]
            if np.isfinite(Mc[k_]).sum() >= 4:
                lines["P-spline"] = self.Bsp @ fm.spline_coefs(Mc[k_][None, :], self.Bsp, self.LAM_S)[0]
            series.append({"label": f"{i} ({POS.get(i, '?')})", "lines": lines})
        store.show(fg.fig_curve_examples(VMID, series, "Raw and smoothed Combine acceleration–speed functions",
                                         f"Rising-phase a(v) per attempt, the athlete mean and its P-spline (λ = {self.LAM_S}, chosen by repeat-attempt agreement). Six athletes evenly spaced in id order.",
                                         "speed [yd/s]", "a_tan [yd/s²]"), "AM12_A_01_raw_and_smoothed_functions", {}, publication=True, source="m12_combine_curves")
        f_full = self.f_full
        if f_full is not None:
            store.show(fg.fig_components(VMID, f_full.mean, f_full.phi, f_full.eig, "Functional principal components of the Combine a(v) curve",
                                         f"Mean ± 2√λ_k φ_k (all eligible athletes, display only; the analysis refits FPCA per training fold). Cumulative variance: {', '.join(f'{x:.2f}' for x in f_full.var_share)}.",
                                         "speed [yd/s]", "a_tan [yd/s²]"), "AM12_A_02_fpca_components", {}, publication=True, source="m12_combine_curves")
        pick = [int(k) for k in np.linspace(0, res.ids.size - 1, 6).astype(int)]
        F = self.FREEZE
        store.show(fg.fig_curve_examples(VMID, [{"label": f"{int(res.ids[k])} ({res.positions[k]})", "lines": {"observed NFL": res.Y[k], "A2 predicted": res.preds["A2"][k], "M1 predicted": res.preds["M1"][k]}} for k in pick],
                                         "Held-out NFL functions: predicted versus observed", f"NFL {F['estimator']} τ = {F['tau']} acceleration–speed function of held-out athletes, with out-of-fold predictions from M1 and A2.",
                                         "speed [yd/s]", "acceleration quantile [yd/s²]"), "AM12_A_03_predicted_vs_observed", {}, publication=True, source="m12a (out-of-fold)")
        store.show(fg.fig_lines(self.pw, "speed_mid", "delta_se", "comparison", "Pointwise transfer effect across speed",
                                "Δ squared error (reference − model) per speed bin, athlete bootstrap 95 % band; positive = the model predicts that part of the NFL function better.",
                                "speed [yd/s]", "Δ squared error [(yd/s²)²]", "lo", "hi", ref=0.0), "AM12_A_04_pointwise_effect", {"pointwise": self.pw}, publication=True, source="m12a_pointwise")
        band = []
        for p in sorted(set(res.positions.tolist())):
            m = res.positions == p
            if m.sum() < C.min_athletes // 2:
                continue
            Yp = res.Y[m]
            bs = np.random.default_rng(C.seed).integers(0, m.sum(), (200, m.sum()))
            with np.errstate(invalid="ignore"):
                mb = np.array([np.nanmean(Yp[b], 0) for b in bs])
                band += [{"position_group": p, "speed": float(VMID[k]), "mean": float(np.nanmean(Yp[:, k])), "lo": float(np.nanquantile(mb[:, k], 0.025)), "hi": float(np.nanquantile(mb[:, k], 0.975))}
                         for k in range(len(self.BINS))]
        band = pl.DataFrame(band)
        store.show(fg.fig_lines(band, "speed", "mean", "position_group", "Functional uncertainty bands by position",
                                "Mean NFL acceleration–speed function per position with athlete-bootstrap 95 % bands (observed game movement, not capacity).",
                                "speed [yd/s]", "acceleration quantile [yd/s²]", "lo", "hi"), "AM12_A_05_functional_bands", {"bands": band}, source="m12c freeze + m11_envelope_hist")
        fp = self.CMP.with_columns((pl.col("mode") + " · " + pl.col("model") + " vs " + pl.col("reference")).alias("label"))
        store.show(fg.fig_forest(fp, "label", "delta_mse", "delta_lo", "delta_hi", "Scalar versus functional representations (held out)",
                                 "Δ athlete-mean squared error of the NFL function (reference − model; positive = model better) on identical athletes; controls shuffle the Combine curves within position or add random scores.",
                                 "Δ MSE [(yd/s²)²]", color_by="mode"), "AM12_A_06_functional_forest", {"compare": fp}, publication=True, source="m12a_compare")
        if self.bypos.height:
            bp = self.bypos.with_columns((pl.col("position_group") + " (n=" + pl.col("n_athletes").cast(pl.String) + ")").alias("label"))
            store.show(fg.fig_forest(bp, "label", "diff", "lo", "hi", "Position-specific comparison: A2 vs A0", "Within-position Δ MSE (A0 − A2), athlete bootstrap; subgroup sensitivity only.", "Δ MSE",
                                     color_by="position_group"), "AM12_A_07_by_position", {"by_position": bp}, source="m12a_by_position")
        if self.pa.height and self.V is not None:
            V, Va, G, g = self.V, self.Va, self.G, self.g
            store.show(fg.fig_curve_examples(g, [{"label": "raw speed–time curves (first 12)", "lines": {f"{j}": V[j] for j in range(min(12, V.shape[0]))}},
                                                 {"label": "elastically aligned", "lines": {f"{j}": Va[j] for j in range(min(12, V.shape[0]))}},
                                                 {"label": "warps γ(t) (phase)", "lines": {f"{j}": G[j] for j in range(min(12, V.shape[0]))}}],
                                             "Phase and amplitude of the forty speed–time curve",
                                             "SRSF dynamic-programming alignment: timing differences move into the warp γ (phase), shape differences remain after alignment (amplitude). Phase is reported, never discarded.",
                                             "time from onset [s]", "speed [yd/s] / warped time [s]"), "AM12_A_08_phase_amplitude", {}, publication=True, source="m12a_phase_amplitude")
        if self.twin_dec.height:
            store.show(fg.fig_scatter(self.twin_dec, "phase_diff_s", "amplitude_diff", "Stopwatch-twin curve decomposition",
                                      "Per twin pair (same position, forty within 0.03 s): difference of mean warps (phase) against difference of aligned curves (amplitude).",
                                      "phase difference ‖γ_a − γ_b‖ [s]", "amplitude difference [yd/s·√s]"), "AM12_A_09_twin_decomposition", {"twins": self.twin_dec}, source="m12a_twin_decomposition")

    @step
    def conclude(self) -> None:
        """Parts 10–12 — comparisons + evidence for 12G, conclusion, documents, run manifest."""
        import polars as pl
        from nflbdb.research.advanced_movement.m12 import advanced_evaluation as ae
        from nflbdb.research.advanced_movement.m12 import config as m12c
        from nflbdb.research.advanced_movement.m12 import reporting as rp
        from nflbdb.research.advanced_movement.m12 import synthesis as sy

        C, rm, CMP, cmp, fpc_rel, el_icc, el_red, rp_sum = self.C, self.controls.run_mode, self.CMP, self.cmp, self.fpc_rel, self.el_icc, self.el_red, self.rp_sum
        comps = [sy.comparison_row("12A", f"{r['mode']}: {r['model']} vs {r['reference']}", "Δ MSE of the NFL function", r["delta_mse"], r["delta_lo"], r["delta_hi"], r["n_athletes"],
                                   m12c.PHYSICAL_TRANSFER, "m12a_compare", rm, experiment=1, primary=(r["mode"], r["model"], r["reference"]) == ("fof", "A2", "A0"))
                 for r in CMP.iter_rows(named=True)]
        if rp_sum:
            comps.append(sy.comparison_row("12A", "hierarchical pooling vs raw (attempt 1 → attempt 2)", "Δ MSE repeat prediction", rp_sum["diff"], rp_sum["lo"], rp_sum["hi"], rp_sum["n_athletes"],
                                           m12c.RELIABILITY, "m12a_hierarchical_repeat", rm, experiment=1))
        COMP = sy.comparison_frame(comps)
        self.W(COMP, "m12a_comparisons")
        ev_rows = []
        for r in cmp.filter(pl.col("reference") == "M1").iter_rows(named=True):
            ev_rows.append(ae.evidence_row(f"12A.{r['model']}", "12A", experiment="1", physical_interpretation=self.MODELS[r["model"]].note or r["model"], input_data="m12_combine_curves + 12C NFL function",
                                           output="NFL acceleration–speed function", n_athletes=r["n_athletes"], delta_rmse=r["delta_mse"], delta_lo=r["delta_lo"], delta_hi=r["delta_hi"],
                                           reliability=fpc_rel["reliability"][0] if r["model"] == "A2" and fpc_rel.height else None,
                                           reliability_lo=fpc_rel["rel_lo"][0] if r["model"] == "A2" and fpc_rel.height else None, compute_cost="minutes",
                                           evaluation_target=m12c.PHYSICAL_TRANSFER, training_regime=m12c.INDUCTIVE, run_mode=rm, config_hash=C.hash()))
        for r in (el_icc.iter_rows(named=True) if el_icc.height else []):
            red = el_red.filter(pl.col("feature") == r["feature"])
            ev_rows.append(ae.evidence_row(f"12A.elastic.{r['feature']}", "12A", physical_interpretation="phase / amplitude of the forty speed–time curve", input_data="m12_combine_speed_time", output=r["feature"],
                                           n_athletes=r["n_athletes"], reliability=r["icc"], reliability_lo=r["icc_lo"], reliability_hi=r["icc_hi"],
                                           stopwatch_r2=red["r2_official"][0] if red.height else None, compute_cost="tens of minutes", evaluation_target=m12c.RELIABILITY, run_mode=rm, config_hash=C.hash()))
        EVID = ae.evidence_frame(ev_rows)
        self.W(EVID, "m12a_evidence")
        prim = COMP.filter(pl.col("primary"))["verdict"].to_list()
        status = rp.status_from(prim)
        facts = {"status": status, "run_mode": rm, "sample": f"{self.ids.size} linked athletes with the shared speed range in both domains; {self.A.shape[0]} forty attempts; {len(self.BINS)} shared bins.",
                 "evidence": [f"{r['mode']} {r['model']} vs {r['reference']}: Δ MSE {rp.ci(r['delta_mse'], r['delta_lo'], r['delta_hi'], 4)} (n={r['n_athletes']}) — {r['verdict']}" for r in CMP.iter_rows(named=True)]
                             + ([f"FPC{r['component']} attempt test–retest {rp.ci(r['reliability'], r['rel_lo'], r['rel_hi'])}" for r in fpc_rel.iter_rows(named=True)] if fpc_rel.height else [])
                             + ([f"elastic {r['feature']}: ICC {rp.ci(r['icc'], r['icc_lo'], r['icc_hi'])}" for r in el_icc.iter_rows(named=True)] if el_icc.height else []),
                 "alternatives": ["The a(v) shape of a standing-start sprint and of game acceleration differ by task; a null functional transfer may reflect task, not athlete.",
                                  "FPCA scores largely re-express the band accelerations: similar performance of A0 and A2 is expected if information is low-dimensional."],
                 "relevance": "Shows whether the whole Combine acceleration profile — not just its summaries — carries information about how the athlete accelerates in games, and at which speeds.",
                 "limitations": ["10 Hz sampling leaves few frames per 0.5 yd/s bin at low speed.", "The NFL function is an upper quantile of observed game movement.", "One cohort."]
                                + ([] if self.opt("elastic") else ["Elastic alignment (exploratory) not run in this run mode."]),
                 "next": ["If transfer concentrates in a speed band, test that band alone on an independent cohort."]}
        text = rp.conclusion_markdown("12A", facts)
        md(text)
        rp.write_conclusion(self.store, "12A", text)
        nl = chr(10)
        rp.update_doc(self.store, "FUNCTIONAL_TRANSFER.md", text + nl + "### Comparisons (identical athletes)" + nl + nl + rp.md_table(CMP, ["mode", "model", "reference", "n_athletes", "delta_mse", "delta_lo", "delta_hi", "verdict"]))
        rp.run_manifest(self.store, "12A", self.ctx.provenance("12A"), ["m12a_compare", "m12a_pointwise", "m12a_fpc_reliability", "m12a_phase_amplitude", "m12a_comparisons", "m12a_evidence"], status,
                        {"freeze_hash": self.FREEZE["freeze_hash"]})
        self.facts, self.conclusion_md, self.m12_status = facts, text, status

    def run_all(self) -> None:
        for s in (self.prepare, self.representations, self.localisation, self.elastic, self.figures, self.conclude):
            s()


__all__ = ["FunctionalTransfer"]
