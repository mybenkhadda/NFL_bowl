"""12C — distributional NFL movement capacity (ported from ``12c_distributional_nfl_movement_capacity.ipynb``).

H1 (measurement): upper conditional acceleration quantiles (τ = 0.90) over the 12.0 shared speed range are more
game-split reliable than the observed maximum (C0a) and the linear envelope (C0b). H2 (transfer): M3 (M1 + fitted
sprint Vmax) lowers held-out RMSE of ``nfl.q90_shared`` vs M1; M2 (M1 + observed top speed) lowers held-out RMSE of
``nfl.game_top_speed_q75`` vs M1; Holm over the two. The estimator behind the target is FROZEN by a declared
reliability rule on NFL data only, before any Combine value is joined.
"""

from __future__ import annotations

from .base import Stage, display_df, md, step


class DistributionalCapacity(Stage):
    track = "12C"
    title = "Distributional NFL movement capacity"
    report_name = "12c_distributional_capacity"
    methodology = (
        "Per athlete, a cube of per-game speed × tangential-acceleration histograms (games = replication unit; athletes with ≥ `min_games_athlete` "
        "games). Estimators of Q_τ(a | v): C0a observed maximum · C0b linear envelope (Milestone 11 rule) · C1 empirical bin quantile · C2 smooth "
        "quantile P-spline (λ tuned once by held-out pinball on a dedicated random game split, seed 999, no Combine value) · C3 position-pooled "
        "empirical Bayes · C4 non-increasing above `shape_vmin` (τ ≥ 0.9). C5 declared optional and NOT implemented. Trait `nfl.q{τ}_shared` = mean "
        "over the shared speed bins, defined only when every shared bin is supported. Reliability: Spearman–Brown split-half over odd/even, three "
        "random and a season split; athlete-bootstrap CIs and paired reliability differences. Held-out forecasts: fit on odd games, score on even "
        "games (pinball, coverage, CRPS, crossing). Target freeze (`dc.freeze_rule` → `m12c_target_freeze.json`, immutable). Transfer: athlete-"
        "grouped outer folds, ridge ladder M0–M6 with inner-fold penalties, paired athlete bootstrap Δ RMSE on identical athletes, Holm over the "
        "two primaries, BH exploratory; within-position athlete-label permutation; by-position subgroups; M1+ sensitivity; C7 replication of the "
        "two 11B high-speed findings under declared variants (all reported).")

    @step
    def prepare(self) -> None:
        """Part 3 — eligibility, memory plan, athlete cubes."""
        import polars as pl
        from nflbdb.research.advanced_movement.m12 import advanced_evaluation as ae
        from nflbdb.research.advanced_movement.m12 import distributional_capacity as dc
        from nflbdb.research.advanced_movement.m12 import foundation as fd

        C, D = self.C, self.D
        self.MAN = fd.check_foundation(self.store, C)
        self.BINS = self.MAN["shared_speed"]["bins"]
        if not self.BINS:
            raise RuntimeError("12.0 found no shared speed range: 12C summary traits are undefined (see m12_shared_speed_support)")
        self.SE_, self.AE_ = ae.speed_edges(C), ae.accel_edges(C)
        self.SHARED = (self.MAN["shared_speed"]["lo"], self.MAN["shared_speed"]["hi"])
        md(f"**Run mode: `{self.controls.run_mode}`** · config `{C.hash()}` · shared speed range {self.SHARED[0]}–{self.SHARED[1]} yd/s ({len(self.BINS)} bins)")
        self.reg = D.m12("m12_cohort_registry")
        self.gts = D.m12("m12_game_top_speed")
        self.off, self.env, self.athf, self.nt = D.officials(), D.envelopes(), D.athlete_features(), D.nfl_targets()
        ids = self.reg.filter(pl.col("elig_12C_nfl"))["nfl_id"].to_list()
        POS = dict(zip(self.gts["nfl_id"].to_list(), self.gts["position_group"].to_list()))
        POS |= {int(i): p for i, p in zip(self.reg["nfl_id"].to_list(), self.reg["position_group"].to_list()) if p is not None}
        self.POS = POS
        n_rows = int(D.hist().filter(pl.col("nfl_id").is_in(ids)).select(pl.len()).collect().item())
        games_mean = float(self.reg.filter(pl.col("elig_12C_nfl"))["nfl_games"].mean() or 0)
        plan = ae.memory_plan([{"item": "histogram rows of eligible athletes", "n_rows": n_rows, "n_cols": 6, "bytes_per_value": 8},
                               {"item": "athlete cubes (games × 23 × 64 int32)", "n_rows": int(len(ids) * games_mean), "n_cols": (self.SE_.size - 1) * (self.AE_.size - 1), "bytes_per_value": 4},
                               {"item": "bootstrap replicates per athlete (transient)", "n_rows": C.env_boot, "n_cols": (self.SE_.size - 1) * (self.AE_.size - 1), "bytes_per_value": 8}],
                              self.knob("memory_budget_gb"))
        display_df(plan)
        h = D.hist_athletes(ids)
        self.cubes = dc.build_cubes(h, POS, C)
        del h
        print(f"{len(self.cubes)} NFL athletes with ≥ {C.min_games_athlete} games · {int(self.reg['elig_12C_transfer'].sum())} linked with a forty")

    @step
    def estimators(self) -> None:
        """Part 4 — C2 penalty on the dedicated tuning split; curves of every estimator × τ on all games."""
        import polars as pl
        from nflbdb.research.advanced_movement.m12 import distributional_capacity as dc

        C = self.C
        lam_tab = dc.tune_lambda(self.cubes, C, self.BINS, C.primary_tau)
        self.LAM = float(lam_tab.filter(pl.col("pinball").is_finite()).sort("pinball", "lambda")["lambda"][0]) if lam_tab["pinball"].is_finite().any() else float(C.spline_lambdas[1])
        self.W(lam_tab, "m12c_lambda_tuning")
        display_df(lam_tab)
        print(f"C2 penalty λ = {self.LAM}")
        self.F_all = dc.curves(self.cubes, None, C.taus, C, self.LAM, C.env_boot, C.seed)
        print("curves for every estimator × τ computed on all games")

    @step
    def athlete_example(self) -> None:
        """Parts 5–6 — one rule-chosen athlete (median frames in the shared range) with every estimator; position quantile curves."""
        import numpy as np
        import polars as pl
        from nflbdb.research.advanced_movement.m12 import advanced_evaluation as ae
        from nflbdb.research.advanced_movement.m12 import distributional_capacity as dc
        from nflbdb.research.advanced_movement.m12 import figures as fg

        C, cubes, store = self.C, self.cubes, self.store
        order = sorted(cubes, key=lambda i: (int(cubes[i].total()[self.BINS].sum()), i))
        pick = order[len(order) // 2]
        cv = {e: self.F_all[C.primary_tau][e][pick] for e in dc.ESTIMATORS}
        store.show(fg.fig_athlete_cloud(cubes[pick].total(), self.SE_, self.AE_, cv, "One athlete: NFL speed–acceleration cloud and estimators",
                                        f"Frames of athlete {pick} ({cubes[pick].position}, {cubes[pick].games.size} games; grey = log frame count). Curves: τ = {C.primary_tau} estimators C1–C4, the observed maximum C0a and the linear envelope C0b. Dotted: shared speed range.", self.SHARED),
                   "AM12_C_01_athlete_cloud", {}, publication=True, source="m11_envelope_hist", n="1 athlete (rule-chosen)")
        pc = []
        for t in C.taus:
            for p in sorted(set(c_.position for c_ in cubes.values())):
                M = np.array([self.F_all[t]["C1_empirical"][i] for i, cb in cubes.items() if cb.position == p])
                if M.shape[0] >= C.min_athletes:
                    with np.errstate(invalid="ignore"):
                        pc += [{"position_group": p, "tau": t, "speed": float(ae.mids(self.SE_)[k]), "accel_q": float(np.nanmean(M[:, k])), "n_athletes": int(np.isfinite(M[:, k]).sum())}
                               for k in range(M.shape[1])]
        pc = pl.DataFrame(pc)
        self.W(pc, "m12c_position_quantile_curves")
        store.show(fg.fig_lines(pc.filter((pl.col("tau") == C.primary_tau) & (pl.col("n_athletes") >= 10)), "speed", "accel_q", "position_group", "Conditional upper-quantile curves by position",
                                f"Mean over athletes of the empirical τ = {C.primary_tau} acceleration quantile per speed bin (bins with ≥ 10 supported athletes). Observed game movement: role and opportunity shape these curves.",
                                "speed [yd/s]", "acceleration quantile [yd/s²]", shade=self.SHARED), "AM12_C_02_position_quantile_curves", {"curves": pc}, publication=True, source="m11_envelope_hist")

    @step
    def reliability(self) -> None:
        """Part 7a — game-split reliability (PRIMARY H1), reliability differences vs C0a / C0b, speed-target reliability."""
        import numpy as np
        import polars as pl
        from nflbdb.research.advanced_movement.m12 import advanced_evaluation as ae
        from nflbdb.research.advanced_movement.m12 import distributional_capacity as dc

        C = self.C
        splits = [("odd_even", 0)] + [("random", s) for s in range(1, C.game_splits + 1)] + [("season", 0)]
        self.rel, self.binrel, self.splitvals = dc.reliability_tables(self.cubes, C, self.BINS, self.LAM, splits, C.seed, progress=lambda s: print("reliability", s, flush=True))
        spd_rel = dc.speed_reliability(self.gts, C, C.seed)
        self.W(self.rel, "m12c_reliability")
        self.W(self.binrel, "m12c_bin_reliability")
        self.W(self.splitvals, "m12c_split_traits")
        self.W(spd_rel, "m12c_speed_reliability")
        rdiff = pl.DataFrame([dc.reliability_difference(self.splitvals, e, ref, C.primary_tau, "odd_even:0", max(200, C.n_boot // 5), C.seed)
                              for e in dc.QUANTILE_ESTIMATORS for ref in ("C0a_observed_max", "C0b_linear_envelope")], infer_schema_length=None)
        rdiff = rdiff.with_columns(pl.struct("lo", "hi").map_elements(lambda s: ae.reading(s["lo"] if s["lo"] is not None else np.nan, s["hi"] if s["hi"] is not None else np.nan),
                                                                      return_dtype=pl.String).alias("verdict"))
        self.rdiff = rdiff
        self.W(rdiff, "m12c_reliability_difference")
        t = self.rel.filter(pl.col("split") == "odd_even:0").select("tau", "estimator", "n_athletes", "reliability", "rel_lo", "rel_hi", "rho_half", "reliability_pos_adj",
                                                                   "shape_agreement").sort("tau", "estimator")
        display_df(t)
        display_df(rdiff)
        display_df(spd_rel)
        self.add_report_table("Game-split reliability (odd / even)", t)
        self.add_report_table("Reliability difference vs the envelope estimators (PRIMARY H1)", rdiff)

    @step
    def forecasts(self) -> None:
        """Part 7b — held-out distributional forecasts: fit on odd games, score on even games."""
        from nflbdb.research.advanced_movement.m12 import distributional_capacity as dc

        C = self.C
        ft = dc.forecast_table(self.cubes, C, self.BINS, self.LAM)
        self.fs = dc.forecast_summary(ft, C, C.seed)
        self.W(ft, "m12c_forecast_athletes")
        self.W(self.fs, "m12c_forecast_summary")
        display_df(self.fs.sort("tau", "estimator"))
        self.add_report_table("Held-out forecast summary (pinball / coverage / CRPS)", self.fs)

    @step
    def freeze(self) -> None:
        """Part 7c — target freeze (declared rule, NFL data only, BEFORE any Combine value is joined); athlete targets."""
        from nflbdb.research.advanced_movement.m12 import distributional_capacity as dc

        C = self.C
        decision = dc.freeze_rule(self.rel, C)
        self.FREEZE = dc.write_freeze(self.store, decision, self.BINS, self.LAM, C)
        F = self.FREEZE
        md(f"**Frozen primary target** `nfl.q90_shared` = estimator **{F['estimator']}** at τ = {F['tau']} over {len(F['shared_bins'])} shared bins "
           f"(freeze `{F['freeze_hash']}`, {F['frozen_at']}){' — FLAG: ' + F['decision']['flag'] if F['decision'].get('flag') else ''}")
        self.tg = dc.athlete_targets(self.cubes, self.gts, self.env, F, C, C.seed)
        self.W(self.tg, "m12c_targets")
        display_df(self.tg.select([c for c in self.tg.columns if c.startswith(("nfl.", "env."))]).describe())

    @step
    def transfer(self) -> None:
        """Part 7d — Combine → NFL transfer ladder (PRIMARY H2) on identical athletes per target."""
        import polars as pl
        from nflbdb.research.advanced_movement.m11 import combine_drills as cd
        from nflbdb.research.advanced_movement.m11 import speed_state as ss
        from nflbdb.research.advanced_movement.m12 import distributional_capacity as dc
        from nflbdb.research.advanced_movement.m12 import foundation as fd
        from nflbdb.research.advanced_movement.m12 import functional_movement as fm
        from nflbdb.research.advanced_movement.m12 import multiview_traits as mv

        C, D, store = self.C, self.D, self.store
        cw = cd.wide(self.athf, ["forty.v_max_obs", "forty.sprint_vmax", "forty.phase_V0"])
        nwide = ss.nfl_wide(self.nt)
        nw = nwide.select("nfl_id", *[c for c in ("nfl.vmax_p90",) if c in nwide.columns])
        fman = D.m12("m12_fold_manifest").select("nfl_id", "fold")
        link_ids = self.reg.filter(pl.col("elig_12C_transfer"))["nfl_id"].to_list()
        self.data = (self.off.filter(pl.col("nfl_id").is_in(link_ids)).join(cw, on="nfl_id", how="left").join(self.tg.drop("position_group"), on="nfl_id", how="inner")
                     .join(nw, on="nfl_id", how="left").join(fman, on="nfl_id", how="inner").sort("nfl_id"))
        curves = D.m12("m12_combine_curves")
        ids_c, Mc, bridged = fm.combine_athlete_matrix(curves, self.BINS, C)
        m5 = lambda a_ids: fm.fpca_block(a_ids, ids_c, Mc, C)                       # noqa: E731
        m6 = None
        if (store.data_dir / "m12d_selected_representation.json").is_file():
            sel = store.read_json("m12d_selected_representation", fd.NB["12D"])
            views = mv.view_features(self.athf["feature"].unique().to_list())
            mv_ids, mv_X, mv_feats, mv_views = mv.athlete_matrix(self.athf, views)
            m6 = lambda a_ids: mv.factor_block(sel["model"], sel["k"], mv_ids, mv_X, mv_views, self.POS, a_ids, C, C.seed)   # noqa: E731
        self.MODELS = dc.ladder(m5, m6)
        targets = ["nfl.q90_shared", "nfl.game_top_speed_q75", "nfl.q95_shared", "nfl.q50_shared", "env.lin_V0", "nfl.vmax_p90"]
        self.tr_rows, tr_oof = dc.transfer(self.data, targets, self.MODELS, C, C.seed)
        self.W(self.tr_rows, "m12c_transfer")
        self.W(tr_oof, "m12c_transfer_oof")
        t = self.tr_rows.select("target", "model", "reference", "primary", "n_athletes", "delta_rmse", "delta_lo", "delta_hi", "delta_pct", "p_holm_primary", "p_bh_exploratory", "verdict")
        display_df(t)
        print("M6 multi-view:", "included" if m6 is not None else "not_run (12D not executed yet; evaluated in 12G)")
        self.add_report_table("Transfer ladder (identical athletes per target)", t)

    @step
    def controls_and_subgroups(self) -> None:
        """Part 7d (cont.) — within-position permutation null, by-position subgroups, M1+ sensitivity."""
        import polars as pl
        from nflbdb.research.advanced_movement.m12 import advanced_evaluation as ae

        C, M = self.C, self.MODELS
        static = ["forty.v_max_obs", "forty.sprint_vmax", "forty.phase_V0"]
        dp = self.data.filter(pl.all_horizontal([pl.col(k).is_not_null() for k in static]) & pl.col("nfl.q90_shared").is_not_null())
        self.perm = ae.permutation_blocks(dp, "nfl.q90_shared", M["M1"], M["M3"], C, C.seed) if self.opt("permutation") and dp.height >= C.min_athletes else {}
        res_p = ae.oof_ladder(dp, "nfl.q90_shared", {"M1": M["M1"], "M3": M["M3"], "M5": M["M5"]}, C, C.seed)
        bypos = pl.concat([ae.compare_by_position(res_p, m, "M1", C, C.seed, target="nfl.q90_shared") for m in ("M3", "M5")], how="diagonal_relaxed")
        plus = ae.oof_ladder(dp, "nfl.q90_shared", {"M1+": ae.Model(list(ae.M1_PLUS)), "M3+": ae.Model([*ae.M1_PLUS, "forty.sprint_vmax"])}, C, C.seed)
        sens_plus = ae.compare(plus, [("M3+", "M1+")], C, C.seed, target="nfl.q90_shared", note="M1 with official agility times")
        self.W(pl.DataFrame([self.perm]) if self.perm else pl.DataFrame(), "m12c_permutation")
        self.W(bypos, "m12c_transfer_by_position")
        self.W(sens_plus, "m12c_transfer_m1plus")
        display_df(pl.DataFrame([self.perm])) if self.perm else md("_permutation not run (options['12C']['permutation'] is off)_")
        display_df(bypos.select("target", "position_group", "model", "n_athletes", "delta_rmse", "delta_lo", "delta_hi", "verdict") if bypos.height else bypos)
        display_df(sens_plus)

    @step
    def replication(self) -> None:
        """Part 7e — C7 replication of the two 11B high-speed findings under declared variants (all reported)."""
        import numpy as np
        import polars as pl
        from nflbdb.research.advanced_movement.m12 import advanced_evaluation as ae
        from nflbdb.research.advanced_movement.m12 import distributional_capacity as dc

        C, D, cubes = self.C, self.D, self.cubes
        rep_tab, rep_sum, rep_rel, rep_pos = pl.DataFrame(), pl.DataFrame(), pl.DataFrame(), pl.DataFrame()
        if self.opt("replication"):
            ep = D.episodes()
            first = ep.group_by("nfl_id").agg(pl.col("season").min().alias("_f"))

            def subset_cubes(which):
                out = {}
                for i, cb in cubes.items():
                    m = cb.seasons == cb.seasons.min() if which == "first" else cb.seasons > cb.seasons.min()
                    if m.sum() >= max(2, C.min_games_athlete // 2):
                        out[i] = dc.Cube(i, cb.position, cb.games[m], cb.seasons[m], cb.H[m])
                return out

            def season_targets(cs, cond):
                t = pl.DataFrame([{"nfl_id": i, "env.lin_V0": dc.linear_envelope_curve(cb.total(), C)[1]["V0"]} for i, cb in cs.items()],
                                 schema={"nfl_id": pl.Int64, "env.lin_V0": pl.Float64})
                e2 = ep.join(first, on="nfl_id").filter(cond)
                v = (e2.filter(pl.col("demand_id") == "sustained_high").group_by("nfl_id").agg(pl.col("v_max").quantile(0.9, "linear").alias("nfl.vmax_p90"), pl.len().alias("_n"))
                     .filter(pl.col("_n") >= 10).drop("_n"))
                return t.join(v, on="nfl_id", how="left")

            base = self.data
            extra = {}
            for nm, which, cond in (("first_season_only", "first", pl.col("season") == pl.col("_f")), ("later_seasons_only", "later", pl.col("season") > pl.col("_f"))):
                extra[nm] = base.drop("env.lin_V0", "nfl.vmax_p90", strict=False).join(season_targets(subset_cubes(which), cond), on="nfl_id", how="inner")
            for k in (16, 24):
                extra[f"min_games_{k}"] = base.filter(pl.col("n_games") >= k)
            rep_tab = dc.replication_table(dc.replication_variants(base, extra), C, C.seed)
            rep_sum = dc.replication_summary(rep_tab)
            va, vb = [], []
            for i, cb in cubes.items():
                a_, b_ = dc.game_splits(cb, "odd_even")
                va.append(dc.linear_envelope_curve(cb.total(a_), C)[1]["V0"])
                vb.append(dc.linear_envelope_curve(cb.total(b_), C)[1]["V0"])
            r1 = ae.split_reliability(np.array(va), np.array(vb), max(200, C.n_boot // 5), C.seed)
            eh = (ep.filter(pl.col("demand_id") == "sustained_high").with_columns((pl.col("game_id").rank("dense").over("nfl_id") % 2).alias("_h"))
                  .group_by("nfl_id", "_h").agg(pl.col("v_max").quantile(0.9, "linear").alias("_v"), pl.len().alias("_n")).filter(pl.col("_n") >= 5)
                  .pivot(on="_h", index="nfl_id", values="_v").drop_nulls())
            r2 = ae.split_reliability(eh[eh.columns[1]].to_numpy(), eh[eh.columns[2]].to_numpy(), max(200, C.n_boot // 5), C.seed) if eh.width == 3 else {"n_athletes": 0}
            rep_rel = pl.DataFrame([{"target": "env.lin_V0", **r1}, {"target": "nfl.vmax_p90", **r2}], infer_schema_length=None)
            for feat, tgt, _ in dc.REPLICATION_PAIRS:
                dd = base.filter(pl.col(feat).is_not_null() & pl.col(tgt).is_not_null())
                rr = ae.oof_ladder(dd, tgt, {"M1": ae.Model(list(ae.M1_COLS)), "M1+f": ae.Model([*ae.M1_COLS, feat])}, C, C.seed)
                rep_pos = pl.concat([rep_pos, ae.compare_by_position(rr, "M1+f", "M1", C, C.seed, feature=feat, target=tgt)], how="diagonal_relaxed")
        else:
            md("_C7 replication not run (options['12C']['replication'] is off): reported as not run, never as a negative result._")
        self.rep_tab, self.rep_sum = rep_tab, rep_sum
        self.W(rep_tab, "m12c_replication")
        self.W(rep_sum, "m12c_replication_summary")
        self.W(rep_rel, "m12c_replication_target_reliability")
        self.W(rep_pos, "m12c_replication_by_position")
        if rep_tab.height:
            display_df(rep_tab.select("feature", "target", "variant", "n_athletes", "delta_rmse", "delta_lo", "delta_hi", "delta_pct", "verdict"))
            display_df(rep_sum)
            display_df(rep_rel)
            display_df(rep_pos)
            self.add_report_table("C7 replication summary", rep_sum)

    @step
    def uncertainty(self) -> None:
        """Part 8 — exposure sensitivity of the primary reliability; game-bootstrap SE of each athlete trait."""
        import numpy as np
        import polars as pl
        from nflbdb.research.advanced_movement.m12 import advanced_evaluation as ae
        from nflbdb.research.advanced_movement.m12 import distributional_capacity as dc

        C, cubes = self.C, self.cubes
        ng = pl.DataFrame({"nfl_id": list(cubes), "n_games": [int(cb.games.size) for cb in cubes.values()]})
        sv = self.splitvals.filter((pl.col("split") == "odd_even:0") & (pl.col("tau") == C.primary_tau)).join(ng, on="nfl_id")
        expo = []
        for k in (C.min_games_athlete, 16, 24):
            for (e,), g in sv.filter(pl.col("n_games") >= k).partition_by(["estimator"], as_dict=True).items():
                expo.append({"min_games": k, "estimator": e, **ae.split_reliability(g["trait_a"].to_numpy(), g["trait_b"].to_numpy(), max(200, C.n_boot // 5), C.seed)})
        expo = pl.DataFrame(expo, infer_schema_length=None)
        self.W(expo, "m12c_exposure_sensitivity")
        unc = []
        for i, cb in cubes.items():
            reps = dc.bootstrap_hists(cb, np.ones(cb.games.size, bool), C.env_boot, C.seed)
            if reps is None:
                continue
            tv = np.array([dc.trait(dc.q_empirical(r, C.primary_tau, C), self.BINS) for r in reps])
            tv = tv[np.isfinite(tv)]
            if tv.size >= 20:
                unc.append({"nfl_id": i, "position_group": cb.position, "n_games": int(cb.games.size), "trait_se": float(tv.std()), "trait_lo": float(np.quantile(tv, 0.025)),
                            "trait_hi": float(np.quantile(tv, 0.975))})
        self.unc = pl.DataFrame(unc)
        self.W(self.unc, "m12c_trait_uncertainty")
        display_df(expo.select("min_games", "estimator", "n_athletes", "reliability", "rel_lo", "rel_hi").sort("min_games", "estimator"))

    @step
    def figures(self) -> None:
        """Part 9 — reliability forest, calibration, reliability by τ, position upper tail, speed correspondence, transfer forest,
        high-speed support, trait uncertainty, negative control, reliability along speed."""
        import numpy as np
        import polars as pl
        from nflbdb.research.advanced_movement.m12 import distributional_capacity as dc
        from nflbdb.research.advanced_movement.m12 import figures as fg

        C, store, rel = self.C, self.store, self.rel
        self.r9 = rel.filter((pl.col("split") == "odd_even:0") & (pl.col("tau") == C.primary_tau)).sort("estimator")
        store.show(fg.fig_forest(self.r9, "estimator", "reliability", "rel_lo", "rel_hi", "Extreme envelope versus robust quantile: game-split reliability",
                                 f"Spearman–Brown reliability (odd / even games, athlete bootstrap 95 % CI) of the athlete trait over the shared speed range, τ = {C.primary_tau}; C0a = observed maximum, C0b = linear envelope.",
                                 "split-half reliability", ref=0.0), "AM12_C_03_envelope_vs_quantile_reliability", {"reliability": self.r9}, publication=True, source="m12c_reliability")
        store.show(fg.fig_calibration(self.fs.filter(pl.col("estimator").is_in(list(dc.ESTIMATORS))), "Held-out-game calibration",
                                      "Share of held-out (even-game) frames at or below each estimator fitted on odd games, averaged over athletes and supported shared bins. Diagonal = calibrated. The observed maximum is not a quantile."),
                   "AM12_C_04_heldout_calibration", {"forecast": self.fs}, publication=True, source="m12c_forecast_summary")
        rq = rel.filter(pl.col("split") == "odd_even:0").select("tau", "estimator", "reliability", "rel_lo", "rel_hi")
        store.show(fg.fig_lines(rq, "tau", "reliability", "estimator", "Reliability by quantile level", "Odd / even game-split reliability of each estimator trait across τ (C0a / C0b do not depend on τ).", "τ", "split-half reliability", "rel_lo", "rel_hi"),
                   "AM12_C_05_reliability_by_quantile", {"reliability": rq}, publication=True, source="m12c_reliability")
        lt = self.tg.unpivot(index=["nfl_id", "position_group"], on=["nfl.q90_shared", "nfl.q95_shared", "nfl.game_top_speed_q75"], variable_name="feature", value_name="value")
        store.show(fg.fig_distributions(lt, "feature", "value", "position_group", ["nfl.q90_shared", "nfl.q95_shared", "nfl.game_top_speed_q75"], "Position-specific upper-tail NFL traits",
                                        "Athlete distribution of the frozen τ = 0.90 / 0.95 shared-range acceleration traits and the game top-speed q75 (n per box in the label).",
                                        {"nfl.q90_shared": "yd/s²", "nfl.q95_shared": "yd/s²", "nfl.game_top_speed_q75": "yd/s"}), "AM12_C_06_position_upper_tail", {"targets": lt}, source="m12c_targets")
        store.show(fg.fig_scatter(self.data.filter(pl.col("forty.sprint_vmax").is_not_null() & pl.col("nfl.game_top_speed_q75").is_not_null()), "forty.sprint_vmax", "nfl.game_top_speed_q75",
                                  "Combine versus NFL speed capacity", "Fitted Combine sprint Vmax against the athlete q75 of per-game NFL top speed (descriptive; colour = position).", "Combine sprint Vmax [yd/s]", "NFL game top speed q75 [yd/s]"),
                   "AM12_C_07_speed_correspondence", {}, publication=True, source="m12c_targets + m11_athlete_features")
        tf = self.tr_rows.filter(pl.col("reference") == "M1").with_columns((pl.col("target") + " · " + pl.col("model") + " (n=" + pl.col("n_athletes").cast(pl.String) + ")").alias("label"))
        store.show(fg.fig_forest(tf, "label", "delta_rmse", "delta_lo", "delta_hi", "Paired held-out predictive improvement over official tests",
                                 "Δ RMSE = RMSE(M1) − RMSE(model) on identical held-out athletes per target (positive = better), paired athlete bootstrap 95 % CI.", "Δ RMSE (target units)", color_by="model"),
                   "AM12_C_08_heldout_transfer", {"transfer": tf}, publication=True, source="m12c_transfer")
        POS = self.POS
        nsup = self.D.m12("m12_nfl_bin_support").join(pl.DataFrame({"nfl_id": list(POS), "position_group": list(POS.values())}), on="nfl_id", how="left")
        hs = (nsup.filter(pl.col("nfl_id").is_in(list(self.cubes))).with_columns((pl.col("n_frames") >= C.min_frames_bin_nfl).alias("ok")).group_by("position_group", "speed_bin")
              .agg(pl.col("ok").sum().alias("n_ok")).join(nsup.filter(pl.col("nfl_id").is_in(list(self.cubes))).group_by("position_group").agg(pl.col("nfl_id").n_unique().alias("n")), on="position_group")
              .with_columns((pl.col("n_ok") / pl.col("n")).alias("share"), (pl.col("speed_bin") * C.speed_step + C.speed_step / 2).alias("speed")).filter(pl.col("n") >= C.min_athletes))
        store.show(fg.fig_lines(hs, "speed", "share", "position_group", "High-speed exposure support", f"Share of eligible athletes per position with ≥ {C.min_frames_bin_nfl} frames in each speed bin. Unsupported high-speed regions are not estimated.",
                                "speed [yd/s]", "share of athletes supported", shade=self.SHARED), "AM12_C_09_high_speed_support", {"support": hs}, source="m12_nfl_bin_support")
        if self.unc.height:
            store.show(fg.fig_scatter(self.unc, "n_games", "trait_se", "Uncertainty of the upper-tail trait", "Game-bootstrap SE of each athlete C1 τ = 0.90 shared-range trait against games observed.", "NFL games", "trait SE [yd/s²]"),
                       "AM12_C_10_trait_uncertainty", {"uncertainty": self.unc}, source="m12c_trait_uncertainty")
        if self.perm:
            p = self.perm
            store.show(fg.fig_null_vs_observed({"athlete-label permutation": np.array([p["null_mean"], p["null_q95"]])}, {"athlete-label permutation": p["observed"]},
                                               "Negative control: permutation null (summary)", f"Observed Δ RMSE of M3 vs M1 against the null mean / 95th percentile of {p['n_perm']} within-position permutations (p = {p['p_value']:.3f}).", "Δ RMSE"),
                       "AM12_C_11_negative_control", {}, source="m12c_permutation")
        br = self.binrel.filter((pl.col("split") == "odd_even:0") & (pl.col("tau") == C.primary_tau)).with_columns((pl.col("speed_bin") * C.speed_step + C.speed_step / 2).alias("speed"))
        store.show(fg.fig_lines(br, "speed", "reliability", "estimator", "Reliability along the speed axis", f"Per speed bin: split-half reliability of each estimator at τ = {C.primary_tau} (athletes supported in both halves).", "speed [yd/s]", "split-half reliability"),
                   "AM12_C_12_bin_reliability", {"bin_reliability": br}, source="m12c_bin_reliability")

    @step
    def conclude(self) -> None:
        """Parts 10–12 — comparisons + evidence for 12G, conclusion, documents, run manifest."""
        import polars as pl
        from nflbdb.research.advanced_movement.m12 import advanced_evaluation as ae
        from nflbdb.research.advanced_movement.m12 import config as m12c
        from nflbdb.research.advanced_movement.m12 import figures as fg
        from nflbdb.research.advanced_movement.m12 import reporting as rp
        from nflbdb.research.advanced_movement.m12 import synthesis as sy

        C, rm, store = self.C, self.controls.run_mode, self.store
        rdiff, fs, tr_rows, rep_tab, rep_sum, rel, FREEZE = self.rdiff, self.fs, self.tr_rows, self.rep_tab, self.rep_sum, self.rel, self.FREEZE
        comps = [sy.comparison_row("12C", f"reliability {r['estimator']} vs {r['reference']} (τ={r['tau']})", "reliability difference", r["diff"], r["lo"], r["hi"], r["n_athletes"],
                                   m12c.RELIABILITY, "m12c_reliability_difference", rm, experiment=3, primary=True) for r in rdiff.iter_rows(named=True)]
        comps += [sy.comparison_row("12C", f"held-out pinball {r['estimator']} vs C1 (τ={r['tau']})", "pinball gain", r["gain_vs_ref"], r["gain_lo"], r["gain_hi"], r["n_athletes"],
                                    m12c.RELIABILITY, "m12c_forecast_summary", rm, experiment=3) for r in fs.filter(pl.col("estimator") != "C1_empirical").iter_rows(named=True)]
        comps += [sy.comparison_row("12C", f"transfer {r['model']} vs {r['reference']} → {r['target']}", "Δ RMSE", r["delta_rmse"], r["delta_lo"], r["delta_hi"], r["n_athletes"],
                                    m12c.PHYSICAL_TRANSFER, "m12c_transfer", rm, primary=bool(r["primary"])) for r in tr_rows.iter_rows(named=True)]
        comps += ([sy.comparison_row("12C", f"replication {r['feature']} → {r['target']} [{r['variant']}]", "Δ RMSE", r.get("delta_rmse"), r.get("delta_lo"), r.get("delta_hi"), r.get("n_athletes"),
                                     m12c.PHYSICAL_TRANSFER, "m12c_replication", rm) for r in rep_tab.filter(pl.col("status") == "computed").iter_rows(named=True)] if rep_tab.height else [])
        COMP = sy.comparison_frame(comps)
        self.W(COMP, "m12c_comparisons")
        ev_rows = []
        for r in rel.filter((pl.col("split") == "odd_even:0") & (pl.col("tau") == C.primary_tau)).iter_rows(named=True):
            ev_rows.append(ae.evidence_row(f"nfl.{r['estimator']}.q90_shared", "12C", experiment="3", physical_interpretation="upper conditional acceleration over the shared speed range",
                                           input_data="m11_envelope_hist", output="athlete trait [yd/s²]", n_athletes=r["n_athletes"], reliability=r["reliability"], reliability_lo=r["rel_lo"],
                                           reliability_hi=r["rel_hi"], compute_cost="minutes", evaluation_target=m12c.RELIABILITY, run_mode=rm, config_hash=C.hash(),
                                           limitations="observed under game opportunity; no play context in the histograms"))
        for r in tr_rows.filter(pl.col("reference") == "M1").iter_rows(named=True):
            ev_rows.append(ae.evidence_row(f"12C.{r['model']}→{r['target']}", "12C", physical_interpretation=self.MODELS[r["model"]].note, input_data="Combine + officials", output=r["target"],
                                           n_athletes=r["n_athletes"], delta_rmse=r["delta_rmse"], delta_lo=r["delta_lo"], delta_hi=r["delta_hi"], compute_cost="seconds",
                                           evaluation_target=m12c.PHYSICAL_TRANSFER, training_regime=m12c.INDUCTIVE, run_mode=rm, config_hash=C.hash()))
        EVID = ae.evidence_frame(ev_rows)
        self.W(EVID, "m12c_evidence")
        h1 = [r["verdict"] for r in rdiff.filter(pl.col("estimator") == FREEZE["estimator"]).iter_rows(named=True)]
        h2 = [r["verdict"] for r in tr_rows.filter(pl.col("primary")).iter_rows(named=True)]
        status = rp.status_from([*h1, *h2])
        SH = self.SHARED
        facts = {"status": status, "run_mode": rm,
                 "sample": f"{len(self.cubes)} NFL athletes (≥ {C.min_games_athlete} games); transfer cohort {self.data.height} linked athletes; shared range {SH[0]}–{SH[1]} yd/s.",
                 "evidence": [f"frozen estimator {FREEZE['estimator']} (τ = {FREEZE['tau']})"]
                             + [f"{r['estimator']} vs {r['reference']}: reliability {rp.fmt(r['rel_a'])} vs {rp.fmt(r['rel_b'])}, Δ {rp.ci(r['diff'], r['lo'], r['hi'])} (n={r['n_athletes']}) — {r['verdict']}" for r in rdiff.iter_rows(named=True)]
                             + [f"{r['model']} vs M1 → {r['target']}: Δ RMSE {rp.ci(r['delta_rmse'], r['delta_lo'], r['delta_hi'], 4)} (n={r['n_athletes']}, Holm p {rp.fmt(r['p_holm_primary'])}) — {r['verdict']}"
                                for r in tr_rows.filter(pl.col("primary")).iter_rows(named=True)]
                             + ([f"replication {r['feature']}: {r['n_variants']} variants, share with Δ CI > 0 = {rp.fmt(r['share_ci_above_0'], 2)}, share Δ > 0 = {rp.fmt(r['share_positive'], 2)}"
                                 for r in rep_sum.iter_rows(named=True)] if rep_sum.height else []),
                 "alternatives": ["Role and play design decide when athletes accelerate: an upper quantile mixes capacity with opportunity.",
                                  "Position and body size are shared causes of Combine and NFL speed; M1 holds them but not every within-position difference.",
                                  "The 11B high-speed gains were selected among many comparisons; a replication share near the noise level is expected under no effect."],
                 "relevance": "Defines whether a reproducible, context-aware NFL movement target exists that can replace unstable maxima, and whether Combine speed predicts it beyond the stopwatch.",
                 "limitations": ["One cohort; held-out gains are exploratory until an independent draft class confirms them.", "No play context (down, alignment, route) in the histograms.",
                                 "C5 (distributional forest) not implemented.", "Shared range excludes unsupported high-speed bins by design."]
                                + ([] if self.opt("permutation") else ["Permutation null not run in this run mode."]) + ([] if self.opt("replication") else ["C7 replication not run in this run mode."]),
                 "next": ["Re-run the frozen target on a later draft class (independent confirmation).", "Condition quantiles on play context where tracking supplies it."]}
        text = rp.conclusion_markdown("12C", facts)
        md(text)
        rp.write_conclusion(store, "12C", text)
        nl = chr(10)
        block = (text + nl + "### Reliability (odd / even games, τ = primary)" + nl + nl + rp.md_table(self.r9, ["estimator", "n_athletes", "reliability", "rel_lo", "rel_hi", "shape_agreement"])
                 + nl + "### Transfer (identical athletes per target)" + nl + nl + rp.md_table(tr_rows, ["target", "model", "reference", "n_athletes", "delta_rmse", "delta_lo", "delta_hi", "p_holm_primary", "verdict"]))
        rp.update_doc(store, "DISTRIBUTIONAL_CAPACITY.md", block)
        rp.run_manifest(store, "12C", self.ctx.provenance("12C"), ["m12c_reliability", "m12c_forecast_summary", "m12c_targets", "m12c_transfer", "m12c_replication", "m12c_comparisons", "m12c_evidence"],
                        status, {"freeze_hash": FREEZE["freeze_hash"], "estimator": FREEZE["estimator"]})
        store.show(fg.fig_conclusion(pl.DataFrame([{"item": "H1 reliability vs envelope", "evidence": "; ".join(h1) or "—", "status": rp.status_from(h1)},
                                                   {"item": "H2 transfer (Holm)", "evidence": "; ".join(h2) or "—", "status": rp.status_from(h2)}]),
                                     "12C — conclusion", "Status of the declared primary comparisons of this run."), "AM12_C_13_conclusion", {}, publication=True, source="m12c_comparisons")
        self.facts, self.conclusion_md, self.m12_status = facts, text, status

    def run_all(self) -> None:
        for s in (self.prepare, self.estimators, self.athlete_example, self.reliability, self.forecasts, self.freeze, self.transfer,
                  self.controls_and_subgroups, self.replication, self.uncertainty, self.figures, self.conclude):
            s()


__all__ = ["DistributionalCapacity"]
