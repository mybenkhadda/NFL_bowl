"""12E — conditional optimal transport matching (``12e_conditional_optimal_transport_matching.ipynb``, module definitions restored).

Do comparisons of whole movement DISTRIBUTIONS recover cross-domain athlete similarity that point features miss?
Primary: conditional OT (E5: per shared speed bin, W1 between accelerating-state acceleration distributions, mean over
bins supported in both domains) gives a higher linked-athlete cross-domain retrieval AUC (own NFL distribution vs
same-position others) than the standardized feature distance (E0), paired over athletes; and E5's AUC interval lies
above 0.5 (``m12c.TRACKS`` success rule: both).

Deviation from the committed notebook (documented in docs/nflbowl/M12_MERGE_MATRIX.md): that notebook replaced the
module definitions by inline ones (E5 weighted by pooled speed mass; E0 on histogram moments) and dropped the
consistency, bootstrap and downstream analyses. This stage uses the DECLARED module definitions
(``movement_transport.conditional_w1`` / ``distance`` / ``feature_vectors``) and the declared secondary analyses. Its run
manifest carries ``definitions = module_v1``; earlier 12E outputs without it are never reused.
"""

from __future__ import annotations

from typing import Any

from .base import Stage, display_df, md, step

DEFINITIONS = "module_v1"


class OptimalTransport(Stage):
    track = "12E"
    title = "Conditional optimal transport matching"
    report_name = "12e_optimal_transport"
    methodology = (
        "One physical grid for both domains: the 12.0 shared speed bins (0.5 yd/s) × acceleration bins merged to 0.5 yd/s² over −4…6 yd/s² "
        "(`movement_transport.Grid`; values outside dropped, not clipped). Comparable state: ACCELERATING (a_tan ≥ `ot_accel_on`) — a standing-"
        "start sprint is compared with game movement only where both accelerate at the same speed. NFL: the athlete's eligible post-snap frames "
        "(12C cubes); Combine: the forty attempt frames (`m12_combine_hist`). Distances: E0 Euclidean on physical features z-scored within "
        "position and domain (`FEATURE_PAIRS`) · E1 Mahalanobis (shrunk pooled covariance) · E2 exact W2 (POT; `not_available` otherwise) · E3 "
        "sliced W2 · E4 debiased Sinkhorn divergence (ε = `ot_eps` × median cost) · E5 conditional W1 (equal weight per shared speed bin "
        "supported in both) · E6 unbalanced Sinkhorn on frame SHARES (opportunity mass). Retrieval: per linked athlete, AUC of its own NFL "
        "distribution among same-position candidates (identical candidate pools for every method; chance 0.5), athlete-bootstrap CIs, paired "
        "athlete bootstrap Δ AUC vs E0. Controls: within-position shuffled identities (null), coarsened acceleration bins, uniform mass on each "
        "distribution's support (tests whether retrieval comes from support / opportunity rather than shape); ε sensitivity of E4. Secondary: "
        "within-domain consistency (NFL odd vs even games; Combine attempt 1 vs 2), per-athlete bootstrap CI of the own distance (attempts and "
        "games resampled), stopwatch twins (E5 distance between twins' NFL distributions ranked among same-position pairs), downstream ladder "
        "M1 vs M1 + distance to the TRAINING-fold position reference (`ot_block`) on the 12C targets. Retrieval measures cross-domain "
        "similarity, never football performance.")

    def _reuse_extra_ok(self, manifest: dict[str, Any]) -> bool:
        return manifest.get("definitions") == DEFINITIONS

    # ================================================================== helpers
    def _pairwise(self, method: str, Q: dict, K: dict, grid: Any = None) -> dict:
        from nflbdb.research.advanced_movement.m11 import config as m11c
        from nflbdb.research.advanced_movement.m12 import movement_transport as mt

        C, POS, grid = self.C, self.POS, grid or self.GRID
        if method == "E0_euclidean":
            fn = lambda a, b: mt.fdist(a, b)                                            # noqa: E731
        elif method == "E1_mahalanobis":
            Minv = self.Minv
            fn = lambda a, b: mt.fdist(a, b, Minv)                                      # noqa: E731
        elif method == "E6_unbalanced":
            fn = lambda P, Q_: mt.unbalanced(P, Q_, grid, C.ot_eps, C.unbalanced_rho, C.ot_iters)   # noqa: E731
        else:
            fn = lambda P, Q_, _m=method: mt.distance(_m, P, Q_, grid, C, seed=C.seed)  # noqa: E731
        prog = m11c.Progress(len(Q), f"12E {method}", every_s=30)
        return mt.pairwise(Q, K, POS, fn, progress=lambda: prog.step())

    @step
    def prepare(self) -> None:
        """Parts 3–4 — cohort, common support, restricted distributions per athlete × state × domain, feature vectors."""
        import numpy as np
        import polars as pl
        from nflbdb.research.advanced_movement.m11 import combine_drills as cd
        from nflbdb.research.advanced_movement.m11 import speed_state as ss
        from nflbdb.research.advanced_movement.m12 import advanced_evaluation as ae
        from nflbdb.research.advanced_movement.m12 import distributional_capacity as dc
        from nflbdb.research.advanced_movement.m12 import foundation as fd
        from nflbdb.research.advanced_movement.m12 import movement_transport as mt

        C, D = self.C, self.D
        MAN = fd.check_foundation(self.store, C)
        self.BINS = MAN["shared_speed"]["bins"]
        self.GRID = GRID = mt.Grid(C, self.BINS)
        self.pot = mt.pot_available()
        md(f"**Run mode: `{self.controls.run_mode}`** · config `{C.hash()}` · transport grid {len(GRID.speed_bins)} speed × {GRID.amid.size} acceleration bins · exact W2 (POT): {self.pot}")
        reg = D.m12("m12_cohort_registry")
        self.off, athf, nt = D.officials(), D.athlete_features(), D.nfl_targets()
        self.POS = POS = {int(i): p for i, p in zip(self.off["nfl_id"].to_list(), self.off["position_group"].to_list())}
        ids = sorted(reg.filter(pl.col("elig_12E"))["nfl_id"].to_list())
        h = D.hist_athletes(ids)
        self.cubes = dc.build_cubes(h, POS, C)
        del h
        self.chist = D.m12("m12_combine_hist").filter(pl.col("nfl_id").is_in(ids))
        n_pairs = sum(sum(1 for j in ids if POS.get(j) == POS.get(i)) for i in ids)
        display_df(ae.memory_plan([{"item": "restricted distributions per athlete × state × domain", "n_rows": len(ids) * 4, "n_cols": GRID.points.shape[0]},
                                   {"item": "Sinkhorn cost matrix (one pair, transient)", "n_rows": GRID.points.shape[0], "n_cols": GRID.points.shape[0]},
                                   {"item": f"same-position pairs per method: {n_pairs:,}", "n_rows": n_pairs, "n_cols": 1}], self.knob("memory_budget_gb")))
        self.COMB = {s: mt.combine_hists(self.chist, GRID, C, "forty", s) for s in ("all", "accelerating")}
        self.NFL = {s: mt.nfl_hists(self.cubes, GRID, s) for s in ("all", "accelerating")}
        self.NFL_SHARES = mt.nfl_hists(self.cubes, GRID, "accelerating", shares=True)
        tot = self.chist.filter(pl.col("family") == "forty").group_by("nfl_id").agg(pl.col("n").sum().alias("_t"))
        tot = dict(zip(tot["nfl_id"].to_list(), tot["_t"].to_list()))
        self.COMB_SHARES = {i: M / max(float(tot.get(i, 0.0)), 1.0) for i, M in self.COMB["accelerating"].items()}
        both = sorted(set(self.COMB["accelerating"]) & set(self.NFL["accelerating"]))
        self.both = [i for i in both if self.COMB["accelerating"][i].sum() > 0 and self.NFL["accelerating"][i].sum() > 0]
        cw = cd.wide(athf, [a for a, _ in mt.FEATURE_PAIRS])
        nwide = ss.nfl_wide(nt)
        nw = nwide.select("nfl_id", *[b for _, b in mt.FEATURE_PAIRS if b in nwide.columns])
        fdata = self.off.select("nfl_id", "position_group").filter(pl.col("nfl_id").is_in(self.both)).join(cw, on="nfl_id", how="left").join(nw, on="nfl_id", how="left")
        self.FC, self.FN = mt.feature_vectors(fdata, POS)
        self.Minv = mt.mahalanobis_metric([*self.FC.values(), *self.FN.values()])
        print(f"{len(self.both)} linked athletes with accelerating-state distributions in both domains")

    @step
    def retrieval(self) -> None:
        """Parts 5–7a (PRIMARY) — cross-domain retrieval AUC per method on identical same-position candidate pools; Δ AUC vs E0."""
        import polars as pl
        from nflbdb.research.advanced_movement.m12 import movement_transport as mt

        C, both = self.C, self.both
        methods = ["E0_euclidean", "E1_mahalanobis", "E3_sliced_w2", "E5_conditional_w1"]
        if self.pot:
            methods.insert(2, "E2_exact_w2")
        if self.opt("heavy_methods"):
            methods += ["E4_sinkhorn", "E6_unbalanced"]
        self.methods = methods
        skipped = [m for m in mt.METHODS if m not in methods]
        if skipped:
            md(f"_Not computed in this run: {', '.join(skipped)} ({'POT not installed → E2 not_available; ' if not self.pot else ''}"
               f"{'E4 / E6 need options[12E][heavy_methods]' if not self.opt('heavy_methods') else ''}). Reported as not run, never substituted._")
        qb = {i: self.COMB["accelerating"][i] for i in both}
        kb = {i: self.NFL["accelerating"][i] for i in both}
        self.DIST, tabs, summ = {}, {}, []
        for m in methods:
            if m in ("E0_euclidean", "E1_mahalanobis"):
                Q, K = {i: self.FC[i] for i in both if i in self.FC}, {i: self.FN[i] for i in both if i in self.FN}
            elif m == "E6_unbalanced":
                Q, K = {i: self.COMB_SHARES[i] for i in both}, {i: self.NFL_SHARES[i] for i in both}
            else:
                Q, K = qb, kb
            self.DIST[m] = self._pairwise(m, Q, K)
            tabs[m] = mt.retrieval_auc(self.DIST[m], both, both, self.POS)
            summ.append(mt.auc_summary(tabs[m], m, C, C.seed))
        self.tabs = tabs
        self.RET = pl.DataFrame(summ, infer_schema_length=None)
        self.ATAB = pl.concat([t.with_columns(pl.lit(m).alias("method")) for m, t in tabs.items() if t.height], how="diagonal_relaxed")
        self.DIFF = pl.DataFrame([{"method": m, "reference": "E0_euclidean", **mt.paired_auc_diff(tabs[m], tabs["E0_euclidean"], C, C.seed)}
                                  for m in methods if m != "E0_euclidean" and tabs[m].height and tabs["E0_euclidean"].height], infer_schema_length=None)
        self.W(self.ATAB, "m12e_retrieval_athletes")
        self.W(self.RET, "m12e_retrieval_summary")
        self.W(self.DIFF, "m12e_auc_diff")
        display_df(self.RET)
        display_df(self.DIFF)
        self.add_report_table("Cross-domain retrieval AUC per method (within position)", self.RET)
        self.add_report_table("Paired Δ AUC vs E0 (PRIMARY: E5 vs E0)", self.DIFF)

    @step
    def negative_controls(self) -> None:
        """Controls — within-position shuffled identities (null), coarsened bins, uniform mass on the support; ε sensitivity (E4)."""
        import numpy as np
        import polars as pl
        from nflbdb.research.advanced_movement.m12 import movement_transport as mt

        C, both, POS = self.C, self.both, self.POS
        rows = []
        if self.opt("controls"):
            for m, D_ in self.DIST.items():
                null = mt.shuffled_null(D_, both, both, POS, int(self.knob("n_perm_null")), C.seed)
                obs = float(self.RET.filter(pl.col("method") == m)["auc"][0])
                rows.append({"method": m, "control": "within_position_identity_shuffle", "auc_observed": obs, "auc_control": float(np.nanmean(null)) if null.size else None,
                             "control_q95": float(np.nanquantile(null, 0.95)) if np.isfinite(null).any() else None, "n_permutations": int(null.size),
                             "p_value": float((1 + np.sum(null >= obs)) / (1 + null.size)) if null.size else None})
            Gc = self.GRID.coarsen()
            qc, kc = mt.combine_hists(self.chist, Gc, C, "forty", "accelerating"), mt.nfl_hists(self.cubes, Gc, "accelerating")
            Dc = self._pairwise("E5_conditional_w1", {i: qc[i] for i in both if i in qc}, {i: kc[i] for i in both if i in kc}, grid=Gc)
            sc = mt.auc_summary(mt.retrieval_auc(Dc, both, both, POS), "E5_conditional_w1", C, C.seed)
            rows.append({"method": "E5_conditional_w1", "control": "coarsened_accel_bins_1yds2", "auc_observed": float(self.RET.filter(pl.col("method") == "E5_conditional_w1")["auc"][0]),
                         "auc_control": sc["auc"], "control_lo": sc["auc_lo"], "control_hi": sc["auc_hi"]})
            qu = {i: (self.COMB["accelerating"][i] > 0).astype(float) for i in both}
            ku = {i: (self.NFL["accelerating"][i] > 0).astype(float) for i in both}
            Du = self._pairwise("E5_conditional_w1", qu, ku)
            su = mt.auc_summary(mt.retrieval_auc(Du, both, both, POS), "E5_conditional_w1", C, C.seed)
            rows.append({"method": "E5_conditional_w1", "control": "uniform_mass_on_support", "auc_observed": float(self.RET.filter(pl.col("method") == "E5_conditional_w1")["auc"][0]),
                         "auc_control": su["auc"], "control_lo": su["auc_lo"], "control_hi": su["auc_hi"]})
        self.CTRL = pl.DataFrame(rows, infer_schema_length=None)
        self.W(self.CTRL, "m12e_controls")
        eps_rows = []
        if self.opt("eps_sensitivity"):
            qb = {i: self.COMB["accelerating"][i] for i in both}
            kb = {i: self.NFL["accelerating"][i] for i in both}
            for e in C.ot_eps_grid:
                D_ = mt.pairwise(qb, kb, POS, lambda P, Q, _e=e: mt.distance("E4_sinkhorn", P, Q, self.GRID, C, eps_rel=_e, seed=C.seed))
                eps_rows.append({"eps_rel": e, **mt.auc_summary(mt.retrieval_auc(D_, both, both, POS), "E4_sinkhorn", C, C.seed)})
        self.EPS = pl.DataFrame(eps_rows, infer_schema_length=None)
        self.W(self.EPS, "m12e_eps_sensitivity")
        display_df(self.CTRL)
        if self.EPS.height:
            display_df(self.EPS)
        self.add_report_table("Negative controls", self.CTRL)

    @step
    def consistency(self) -> None:
        """Secondary — within-domain consistency: NFL odd vs even games; Combine attempt 1 vs attempt 2 (does a distribution identify its athlete at all?)."""
        import polars as pl
        from nflbdb.research.advanced_movement.m12 import distributional_capacity as dc
        from nflbdb.research.advanced_movement.m12 import movement_transport as mt

        C, POS, GRID = self.C, self.POS, self.GRID
        rows = []
        self.comb_att, self.att_of = {}, {}
        if self.opt("consistency") or self.opt("own_distance_bootstrap"):
            self.comb_att = mt.combine_hists(self.chist, GRID, C, "forty", "accelerating", by="seq_id")
            att = self.chist.filter(pl.col("family") == "forty").select("nfl_id", "seq_id").unique()
            self.att_of = {int(k[0]): g["seq_id"].to_list() for k, g in att.partition_by(["nfl_id"], as_dict=True).items()}
        if self.opt("consistency"):
            methods = [m for m in self.knob("consistency_methods") if m in self.methods and m not in ("E0_euclidean", "E1_mahalanobis", "E6_unbalanced")]
            sa, sb = {}, {}
            for i, cb in self.cubes.items():
                a_, b_ = dc.game_splits(cb, "odd_even")
                sa[i], sb[i] = a_, b_
            Na, Nb = mt.nfl_hists(self.cubes, GRID, "accelerating", sel=sa), mt.nfl_hists(self.cubes, GRID, "accelerating", sel=sb)
            nfl_ids = sorted(i for i in Na if Na[i].sum() > 0 and Nb.get(i) is not None and Nb[i].sum() > 0)
            Qc, Kc = {}, {}
            if self.store.has("m12_combine_quality"):
                q = self.D.m12("m12_combine_quality").filter((pl.col("family") == "forty") & pl.col("nfl_id").is_in(list(self.cubes)))
                for (i,), g in q.partition_by(["nfl_id"], as_dict=True).items():
                    s = g.sort("attempt_order")["seq_id"].to_list()
                    s = [x for x in s if x in self.comb_att and self.comb_att[x].sum() > 0]
                    if len(s) >= 2:
                        Qc[int(i)], Kc[int(i)] = self.comb_att[s[0]], self.comb_att[s[1]]
            comb_ids = sorted(Qc)
            for m in methods:
                tN = mt.retrieval_auc(self._pairwise(m, {i: Na[i] for i in nfl_ids}, {i: Nb[i] for i in nfl_ids}), nfl_ids, nfl_ids, POS)
                rows.append({**mt.auc_summary(tN, m, C, C.seed), "domain": "nfl", "split": "odd_vs_even_games"})
                if comb_ids:
                    tC = mt.retrieval_auc(self._pairwise(m, Qc, Kc), comb_ids, comb_ids, POS)
                    rows.append({**mt.auc_summary(tC, m, C, C.seed), "domain": "combine", "split": "attempt1_vs_attempt2"})
            if not comb_ids:
                print("Combine attempt consistency skipped: needs m12_combine_quality and ≥ 2 forty attempts per athlete")
        else:
            md("_Within-domain consistency not run (options['12E']['consistency'] is off)._")
        self.CONS = pl.DataFrame(rows, infer_schema_length=None)
        self.W(self.CONS, "m12e_consistency")
        if self.CONS.height:
            display_df(self.CONS)
            self.add_report_table("Within-domain consistency (identification within a domain)", self.CONS)

    @step
    def uncertainty(self) -> None:
        """Secondary — per-athlete bootstrap CI of the own cross-domain E5 distance (attempts and games resampled)."""
        import polars as pl
        from nflbdb.research.advanced_movement.m12 import movement_transport as mt

        C = self.C
        rows = []
        if self.opt("own_distance_bootstrap"):
            for i in self.both[: int(self.knob("boot_athletes"))]:
                rows.append(mt.own_distance_ci(i, self.comb_att, self.att_of, self.cubes.get(i), self.GRID, "E5_conditional_w1", C, C.seed, state="accelerating"))
        else:
            md("_Own-distance bootstrap not run (options['12E']['own_distance_bootstrap'] is off)._")
        self.BOOT = pl.DataFrame(rows, infer_schema_length=None)
        self.W(self.BOOT, "m12e_bootstrap")
        if self.BOOT.height:
            display_df(self.BOOT.describe())

    @step
    def twins(self) -> None:
        """Secondary — stopwatch twins (same position, near-identical official forty): how different are their NFL / Combine distributions
        compared with random same-position pairs? (descriptive; percentile 0.5 = like a random pair)."""
        import numpy as np
        import polars as pl
        from nflbdb.research.advanced_movement.m12 import movement_transport as mt

        C, POS, GRID = self.C, self.POS, self.GRID
        rows = []
        if self.opt("twins"):
            try:
                tw = self.D.twins()
            except FileNotFoundError:
                tw = pl.DataFrame()
                print("stopwatch twins not available: skipped")
            bset = set(self.both)
            cache: dict[str, np.ndarray] = {}

            def ref_pairs(p: str, dom: str) -> np.ndarray:
                k = f"{p}|{dom}"
                if k not in cache:
                    mem = [i for i in self.both if POS.get(i) == p]
                    H = self.NFL["accelerating"] if dom == "nfl" else self.COMB["accelerating"]
                    cache[k] = np.array([mt.distance("E5_conditional_w1", H[a], H[b], GRID, C) for x, a in enumerate(mem) for b in mem[x + 1:]], float)
                return cache[k]

            for r in tw.iter_rows(named=True):
                a, b = int(r["nfl_id_a"]), int(r["nfl_id_b"])
                if a not in bset or b not in bset:
                    continue
                p = POS.get(a)
                dn = mt.distance("E5_conditional_w1", self.NFL["accelerating"][a], self.NFL["accelerating"][b], GRID, C)
                dcmb = mt.distance("E5_conditional_w1", self.COMB["accelerating"][a], self.COMB["accelerating"][b], GRID, C)
                rn, rc = ref_pairs(p, "nfl"), ref_pairs(p, "combine")
                rn, rc = rn[np.isfinite(rn)], rc[np.isfinite(rc)]
                rows.append({"pair_id": r["pair_id"], "position_group": p, "d_forty": r.get("d_forty"), "d_nfl_e5": dn, "d_combine_e5": dcmb,
                             "pct_rank_nfl": float((rn < dn).mean()) if rn.size and np.isfinite(dn) else None,
                             "pct_rank_combine": float((rc < dcmb).mean()) if rc.size and np.isfinite(dcmb) else None, "n_reference_pairs": int(rn.size)})
        self.TWINS = pl.DataFrame(rows, infer_schema_length=None)
        self.W(self.TWINS, "m12e_twins")
        if self.TWINS.height:
            display_df(self.TWINS.select("pct_rank_nfl", "pct_rank_combine").describe())
            self.add_report_table("Stopwatch twins: percentile of their E5 distance among same-position pairs", self.TWINS.select("pct_rank_nfl", "pct_rank_combine").describe())

    @step
    def downstream(self) -> None:
        """Secondary — held-out value: M1 vs M1 + distance of the Combine distribution to the TRAINING-fold position NFL reference."""
        import polars as pl
        from nflbdb.research.advanced_movement.m12 import advanced_evaluation as ae
        from nflbdb.research.advanced_movement.m12 import foundation as fd
        from nflbdb.research.advanced_movement.m12 import movement_transport as mt

        C, D, store = self.C, self.D, self.store
        self.DOWN = pl.DataFrame()
        if not self.opt("downstream"):
            md("_Downstream ladder not run (options['12E']['downstream'] is off)._")
        elif not store.has("m12c_targets"):
            md("_Downstream ladder not run: needs the 12C targets (run 12C first)._")
        else:
            reg = D.m12("m12_cohort_registry")
            tg = store.read_frame("m12c_targets", fd.NB["12C"]).select("nfl_id", "nfl.q90_shared", "nfl.game_top_speed_q75")
            data = (self.off.filter(pl.col("nfl_id").is_in(reg.filter(pl.col("linked"))["nfl_id"].to_list())).filter(pl.col("nfl_id").is_in(self.both))
                    .join(tg, on="nfl_id", how="inner").join(D.m12("m12_fold_manifest").select("nfl_id", "fold"), on="nfl_id", how="inner").sort("nfl_id"))
            comb, nfl, POS, GRID = self.COMB["accelerating"], self.NFL["accelerating"], self.POS, self.GRID
            MODELS = {"M1": ae.Model(list(ae.M1_COLS)), "M1+OT": ae.Model(list(ae.M1_COLS), factory=lambda a: mt.ot_block(a, comb, nfl, POS, GRID, C))}
            tabs = []
            for t in ("nfl.q90_shared", "nfl.game_top_speed_q75"):
                res = ae.oof_ladder(data, t, MODELS, C, C.seed)
                tabs.append(ae.compare(res, [("M1+OT", "M1")], C, C.seed, target=t))
            self.DOWN = pl.concat([t for t in tabs if t.height], how="diagonal_relaxed") if tabs else pl.DataFrame()
        self.W(self.DOWN, "m12e_transfer")
        if self.DOWN.height:
            t = self.DOWN.select("target", "model", "reference", "n_athletes", "delta_rmse", "delta_lo", "delta_hi", "verdict")
            display_df(t)
            self.add_report_table("Downstream ladder (M1 + OT distance)", t)

    @step
    def figures(self) -> None:
        """Parts 8–9 — density maps, conditional distributions, transport plan, distance heatmap, retrieval forest, bootstrap CIs,
        position subgroups, consistency."""
        import numpy as np
        import polars as pl
        from nflbdb.research.advanced_movement.m12 import figures as fg
        from nflbdb.research.advanced_movement.m12 import movement_transport as mt

        C, store, GRID, POS, both = self.C, self.store, self.GRID, self.POS, self.both
        spd_lbl = [f"{v:.2f}" for v in GRID.vmid]
        acc_lbl = [f"{a:.1f}" for a in GRID.amid]
        mc = np.mean([mt.normalise(self.COMB["all"][i]) for i in both], axis=0)
        mn = np.mean([mt.normalise(self.NFL["all"][i]) for i in both], axis=0)
        store.show(fg.fig_matrix(np.log10(np.maximum(mc, 1e-6)), spd_lbl, acc_lbl, "Combine forty movement distribution (density map)",
                                 "Mean normalised share of forty frames per speed × acceleration cell, linked athletes, log10 scale.", "log10 share", cmap="Blues", center=False, annotate=False),
                   "AM12_E_01_density_combine", {}, publication=True, source="m12_combine_hist")
        store.show(fg.fig_matrix(np.log10(np.maximum(mn, 1e-6)), spd_lbl, acc_lbl, "NFL game movement distribution (density map)",
                                 "Mean normalised share of eligible post-snap NFL frames per cell, same athletes, same grid, log10 scale.", "log10 share", cmap="Oranges", center=False, annotate=False),
                   "AM12_E_02_density_nfl", {}, publication=True, source="m11_envelope_hist")
        pick = both[len(both) // 2]
        P, Q = self.COMB["accelerating"][pick], self.NFL["accelerating"][pick]
        cond = []
        for k in range(P.shape[0]):
            for dom, M_ in (("Combine", P), ("NFL", Q)):
                tot = float(M_[k].sum())
                if tot <= 0:
                    continue
                cdf = np.cumsum(M_[k]) / tot
                cond += [{"curve": f"{dom} · v={GRID.vmid[k]:.2f}", "accel": float(a), "cdf": float(c)} for a, c in zip(GRID.amid, cdf)]
        cond = pl.DataFrame(cond) if cond else pl.DataFrame(schema={"curve": pl.String, "accel": pl.Float64, "cdf": pl.Float64})
        if cond.height:
            curves = sorted(cond["curve"].unique().to_list())
            keepc = curves[:: max(1, int(np.ceil(len(curves) / 6)))]
            store.show(fg.fig_lines(cond.filter(pl.col("curve").is_in(keepc)), "accel", "cdf", "curve", "Conditional acceleration distributions (one athlete)",
                                    f"Accelerating-state tangential-acceleration CDFs within shared speed bins for the rule-chosen athlete {pick} ({POS.get(pick)}; median of the cohort in id order).",
                                    "tangential acceleration [yd/s²]", "cumulative share"), "AM12_E_03_conditional_distributions", {"conditional": cond}, publication=True, source="m12e")
        a, b = mt.normalise(P).ravel(), mt.normalise(Q).ravel()
        ma, mb = a > 0, b > 0
        if ma.any() and mb.any():
            eps = C.ot_eps * float(np.median(mt.cost_matrix(GRID.points)))
            _, plan = mt.sinkhorn(a[ma], b[mb], mt.cost_matrix(GRID.points, ma, mb), eps, C.ot_iters)
            store.show(fg.fig_transport_plan(plan, GRID.points[ma], GRID.points[mb], "Transport plan: Combine → NFL (one athlete)",
                                             f"Largest-mass couplings of the entropic transport plan for athlete {pick}; ε = {C.ot_eps} × median cost (the E4 definition)."),
                       "AM12_E_04_transport_plan", {}, publication=True, source="m12e")
        pp = [i for i in both if POS.get(i) == POS.get(pick)][:30]
        D5 = self.DIST["E5_conditional_w1"]
        Mh = np.array([[D5.get((i, j), np.nan) for j in pp] for i in pp], float)
        store.show(fg.fig_matrix(Mh, [str(i) for i in pp], [str(j) for j in pp], f"Cross-domain conditional W1 ({POS.get(pick)})",
                                 "Rows: Combine athlete; columns: NFL athlete (first 30 of the position, id order). The diagonal is each athlete's own distance (E5, module definition).",
                                 "conditional W1 [yd/s²]", cmap="viridis", center=False, annotate=False), "AM12_E_05_distance_heatmap", {}, source="m12e_retrieval_athletes")
        parts = [self.RET.select("method", "auc", "auc_lo", "auc_hi", "n_athletes")]
        if self.CTRL.height and "control_lo" in self.CTRL.columns:
            parts.append(self.CTRL.filter(pl.col("control") != "within_position_identity_shuffle").select(
                pl.concat_str([pl.col("method"), pl.lit(" · "), pl.col("control")]).alias("method"), pl.col("auc_control").alias("auc"), pl.col("control_lo").alias("auc_lo"),
                pl.col("control_hi").alias("auc_hi"), pl.lit(None, dtype=pl.Int64).alias("n_athletes")))
        rf = pl.concat(parts, how="diagonal_relaxed")
        null_mean = float(self.CTRL.filter(pl.col("control") == "within_position_identity_shuffle")["auc_control"].mean()) if self.CTRL.height else None
        store.show(fg.fig_forest(rf, "method", "auc", "auc_lo", "auc_hi", "Feature distance versus transport: cross-domain retrieval",
                                 f"Mean linked-athlete retrieval AUC within position (athlete bootstrap 95 % CI); 0.5 = chance{f'; shuffled-identity null mean {null_mean:.3f}' if null_mean is not None else ''}. Retrieval is not football performance.",
                                 "retrieval AUC", ref=0.5), "AM12_E_06_feature_vs_transport", {"retrieval": rf}, publication=True, source="m12e_retrieval_summary")
        if self.BOOT.height and "d_lo" in self.BOOT.columns:
            bb = self.BOOT.filter(pl.col("d_lo").is_not_null()).with_columns(pl.col("nfl_id").cast(pl.String).alias("label")).head(40)
            if bb.height:
                store.show(fg.fig_forest(bb, "label", "d_median", "d_lo", "d_hi", "Bootstrap uncertainty of the own cross-domain distance",
                                         "E5 distance between each athlete's Combine and NFL accelerating-state distributions with attempts and games resampled; 95 % bootstrap interval.", "E5 distance [yd/s²]"),
                           "AM12_E_07_bootstrap_ci", {"bootstrap": bb}, source="m12e_bootstrap")
        bp = self.ATAB.group_by("method", "position_group").agg(pl.col("auc").mean(), pl.len().alias("n")).filter(pl.col("n") >= max(1, C.min_athletes // 2))
        if bp.height:
            store.show(fg.fig_heatmap(bp, "method", "position_group", "auc", "Retrieval by position subgroup", "Mean cross-domain retrieval AUC per method and sufficiently represented position (subgroup sensitivity).",
                                      fmt="{:.2f}", cmap="RdBu_r"), "AM12_E_08_position_subgroups", {"subgroups": bp}, source="m12e_retrieval_athletes")
        if self.CONS.height:
            cs = self.CONS.with_columns((pl.col("method") + " · " + pl.col("domain")).alias("label"))
            store.show(fg.fig_forest(cs, "label", "auc", "auc_lo", "auc_hi", "Within-domain consistency", "Retrieval of an athlete's own second half (NFL: odd vs even games; Combine: attempt 1 vs 2) among same-position athletes; 0.5 = chance.",
                                     "retrieval AUC", ref=0.5), "AM12_E_09_within_domain_consistency", {"consistency": cs}, source="m12e_consistency")

    @step
    def conclude(self) -> None:
        """Parts 10–12 — comparisons + evidence for 12G (primary = the declared success rule), conclusion, documents, run manifest."""
        import numpy as np
        import polars as pl
        from nflbdb.research.advanced_movement.m12 import advanced_evaluation as ae
        from nflbdb.research.advanced_movement.m12 import config as m12c
        from nflbdb.research.advanced_movement.m12 import reporting as rp
        from nflbdb.research.advanced_movement.m12 import synthesis as sy

        C, rm, RET, DIFF, CONS, DOWN, CTRL = self.C, self.controls.run_mode, self.RET, self.DIFF, self.CONS, self.DOWN, self.CTRL
        sh = lambda x: float(x) - 0.5 if x is not None and np.isfinite(float(x)) else None    # noqa: E731
        comps = [sy.comparison_row("12E", f"retrieval AUC {r['method']} vs {r['reference']}", "Δ retrieval AUC", r["diff"], r["lo"], r["hi"], r["n_athletes"], m12c.PHYSICAL_TRANSFER,
                                   "m12e_auc_diff", rm, experiment=5, primary=r["method"] == "E5_conditional_w1") for r in DIFF.iter_rows(named=True)]
        comps += [sy.comparison_row("12E", f"retrieval AUC {r['method']} − 0.5", "AUC − 0.5", sh(r["auc"]), sh(r["auc_lo"]), sh(r["auc_hi"]), r["n_athletes"], m12c.PHYSICAL_TRANSFER,
                                    "m12e_retrieval_summary", rm, experiment=5, primary=r["method"] == "E5_conditional_w1") for r in RET.iter_rows(named=True)]
        comps += [sy.comparison_row("12E", f"consistency {r['method']} ({r['domain']}, {r['split']}) − 0.5", "AUC − 0.5", sh(r["auc"]), sh(r["auc_lo"]), sh(r["auc_hi"]), r["n_athletes"],
                                    m12c.RELIABILITY, "m12e_consistency", rm, experiment=5) for r in (CONS.iter_rows(named=True) if CONS.height else [])]
        comps += [sy.comparison_row("12E", f"downstream M1+OT vs M1 → {r['target']}", "Δ RMSE", r["delta_rmse"], r["delta_lo"], r["delta_hi"], r["n_athletes"], m12c.PHYSICAL_TRANSFER,
                                    "m12e_transfer", rm, experiment=5, training_regime=m12c.INDUCTIVE) for r in (DOWN.iter_rows(named=True) if DOWN.height else [])]
        COMP = sy.comparison_frame(comps)
        self.W(COMP, "m12e_comparisons")
        rows = []
        for r in RET.iter_rows(named=True):
            cr = CONS.filter((pl.col("method") == r["method"]) & (pl.col("domain") == "nfl")) if CONS.height else CONS
            dr = DOWN.filter(pl.col("target") == "nfl.q90_shared") if DOWN.height and r["method"] == "E5_conditional_w1" else pl.DataFrame()
            dn = dr.row(0, named=True) if dr.height else {}
            rows.append(ae.evidence_row(f"12E.{r['method']}", "12E", experiment="5", physical_interpretation="distribution distance under comparable (accelerating, shared-speed) states",
                                        input_data="m12_combine_hist + m11_envelope_hist", output="retrieval auc", n_athletes=r["n_athletes"], cross_domain_stat=r["auc"],
                                        cross_domain_lo=r["auc_lo"], cross_domain_hi=r["auc_hi"], reliability=cr["auc"][0] if cr.height else None,
                                        delta_rmse=dn.get("delta_rmse"), delta_lo=dn.get("delta_lo"), delta_hi=dn.get("delta_hi"),
                                        compute_cost="minutes" if r["method"] not in ("E4_sinkhorn", "E6_unbalanced") else "tens of minutes", evaluation_target=m12c.PHYSICAL_TRANSFER,
                                        run_mode=rm, config_hash=C.hash()))
        EVID = ae.evidence_frame(rows)
        self.W(EVID, "m12e_evidence")
        status = rp.status_from(COMP.filter(pl.col("primary"))["verdict"].to_list())
        null = CTRL.filter(pl.col("control") == "within_position_identity_shuffle") if CTRL.height else CTRL
        facts = {"status": status, "run_mode": rm,
                 "sample": f"{len(self.both)} linked athletes with accelerating-state distributions in both domains; grid {len(self.GRID.speed_bins)} speed × {self.GRID.amid.size} acceleration bins.",
                 "evidence": [f"{r['method']}: AUC {rp.ci(r['auc'], r['auc_lo'], r['auc_hi'])} (n={r['n_athletes']})" for r in RET.iter_rows(named=True)]
                             + [f"{r['method']} − {r['reference']}: Δ AUC {rp.ci(r['diff'], r['lo'], r['hi'])} (n={r['n_athletes']})" for r in DIFF.iter_rows(named=True)]
                             + [f"null ({r['method']}): mean {rp.fmt(r['auc_control'])}, q95 {rp.fmt(r.get('control_q95'))}, p {rp.fmt(r.get('p_value'))}" for r in null.iter_rows(named=True)]
                             + [f"control {r['control']}: E5 AUC {rp.ci(r['auc_control'], r.get('control_lo'), r.get('control_hi'))}" for r in (CTRL.filter(pl.col("control") != "within_position_identity_shuffle").iter_rows(named=True) if CTRL.height else [])]
                             + [f"consistency {r['method']} {r['domain']}: AUC {rp.ci(r['auc'], r['auc_lo'], r['auc_hi'])}" for r in (CONS.iter_rows(named=True) if CONS.height else [])]
                             + [f"downstream M1+OT → {r['target']}: Δ RMSE {rp.ci(r['delta_rmse'], r['delta_lo'], r['delta_hi'], 4)} — {r['verdict']}" for r in (DOWN.iter_rows(named=True) if DOWN.height else [])],
                 "alternatives": ["Role, position and movement-state exposure shape both distributions: retrieval within position limits but does not remove this.",
                                  "If the uniform-mass control retrieves as well as E5, similarity comes from which states were observed (support), not from how the athlete accelerates in them."],
                 "relevance": "Evaluates whether movement-distribution similarity can identify linked athletes across domains — descriptive matching, not football performance prediction.",
                 "limitations": ["Accelerating state only; the Combine forty covers few speed bins at high speed.", "Repeated candidate sharing can make athlete-bootstrap intervals slightly optimistic.", "One cohort."]
                                + [f"Not computed in this run: {', '.join(m for m in ('E2_exact_w2', 'E4_sinkhorn', 'E6_unbalanced') if m not in self.methods)}."] * any(m not in self.methods for m in ("E2_exact_w2", "E4_sinkhorn", "E6_unbalanced")),
                 "next": ["If E5 identifies athletes beyond the uniform-mass control, test whether the matched NFL distribution adds held-out prediction on an independent cohort."]}
        text = rp.conclusion_markdown("12E", facts)
        md(text)
        rp.write_conclusion(self.store, "12E", text)
        nl = chr(10)
        rp.update_doc(self.store, "OPTIMAL_TRANSPORT.md", text + nl + nl + "### Retrieval" + nl + nl + rp.md_table(RET) + nl + nl + rp.md_table(DIFF))
        rp.run_manifest(self.store, "12E", self.ctx.provenance("12E"), ["m12e_retrieval_athletes", "m12e_retrieval_summary", "m12e_auc_diff", "m12e_controls", "m12e_consistency",
                                                                        "m12e_bootstrap", "m12e_transfer", "m12e_comparisons", "m12e_evidence"], status,
                        {"definitions": DEFINITIONS, "methods": self.methods})
        self.facts, self.conclusion_md, self.m12_status = facts, text, status

    def run_all(self) -> None:
        for s in (self.prepare, self.retrieval, self.negative_controls, self.consistency, self.uncertainty, self.twins, self.downstream, self.figures, self.conclude):
            s()


__all__ = ["DEFINITIONS", "OptimalTransport"]
