"""12F — physics-informed shared movement encoder, run through ``nflbowl.Experiment`` on the portable bundle.

The 12F notebook trained with its own loop on the warehouse sequence store. ``nflbowl.Experiment`` runs the SAME
``se.batch_loss`` objective, splits and estimands on the byte-identical bundle copy of that store (see
``docs/nflbowl/REPRODUCIBILITY.md``), with GPU-resident data, explicit precision (part of the checkpoint signature),
per-epoch resumable checkpoints and fp32 contrastive / InfoNCE / MMD regions. Phases are separate: ``prepare`` →
``train_*`` → ``embed_and_evaluate`` → ``transfer`` → ``collect`` (copies into RUN_DIR) → ``publish_to_warehouse``
(so 12G can read 12F) → ``conclude``.

Two regimes, two tables, never one leaderboard: TRANSDUCTIVE encoders (all sequences, no identity / outcome) give
label-free diagnostics only; INDUCTIVE encoders (one per outer fold, held-out athletes removed from BOTH domains) are
the only ones that enter NFL prediction. Only F4 is trained by default.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .artifacts import STATUS_DONE, STATUS_FAILED, STATUS_SKIPPED, ResearchRun
from .stages.base import display_df, md

TRACK = "12F"


class EncoderSection:
    """The 12F section. ``ctx`` (warehouse) is optional: only used to publish 12F results for 12G."""

    methodology = (
        "Unified sequence store (bundle `sequence_store/`: N × 64 frames × 6 channels speed, a_tan, a_n, yaw, dt, heading_valid; Combine drill "
        "attempts and NFL movement-demand windows). Baselines: F0 engineered (17 masked summary statistics), the random frozen F4 architecture, "
        "F1 TCN-B (Milestone 06, Combine only; when in the bundle). Encoders F2–F6 + ablations (registry `nflbowl.models.registry`); primary F4 = "
        "masked reconstruction + physics heads + temporal contrastive + matched-speed-state cross-domain alignment. Batches are domain-balanced "
        "(half Combine, half NFL). Early stopping on a validation share of ATHLETES; wall-clock cap per encoder (flagged). Label-free evaluation "
        "(standard level = exact research protocol): cross-domain athlete retrieval AUC within position, within-domain retrieval, domain-classifier "
        "AUC and RBF-MMD² (separability is not failure), athlete-grouped ridge probes to physical scalars, shuffled-identity null. Inductive: "
        "retrieval of held-out athletes per fold and the ladder M1 · M1+F0 · M1+F1 · M1+encoder (≤ 8 training-fold PCA dims) on the 12C frozen "
        "target `nfl.q90_shared` (paired athlete bootstrap).")

    def __init__(self, run: ResearchRun, ucfg: Any, sources: Any, ctx: Any = None):
        self.run, self.ucfg, self.sources, self.ctx = run, ucfg, sources, ctx
        self.exp = None
        self.state, self.reason = "pending", ""
        self.trained: list[dict[str, Any]] = []
        self.metrics = self.inductive = self.transfer_table = self.training_table = None
        self.inductive_models: list[str] = []
        self.facts: dict[str, Any] | None = None
        self.conclusion_md: str | None = None
        self.report_tables: list[tuple[str, Any, Any]] = []
        self.track, self.title, self.report_name, self.m12_status, self.missing_optional = TRACK, "Physics-informed shared movement encoder", "12f_shared_encoder", None, []
        self._gate()

    # ================================================================== gate / lifecycle
    def _gate(self) -> None:
        from ..devices import torch_available

        if not self.ucfg.enabled(TRACK):
            self.state, self.reason = "disabled", "disabled in CFG['stages']"
        elif self.sources.bundle_dir is None and not (self.ucfg.build_bundle_from_warehouse and self.sources.warehouse):
            self.state, self.reason = "skipped", "no data bundle (Mode A input) and build_bundle_from_warehouse is off"
        elif not torch_available():
            self.state, self.reason = "skipped", "PyTorch is not importable (12F is not feasible here; 12G proceeds without it)"

    @property
    def active(self) -> bool:
        return self.state in ("pending", "running")

    def explain(self) -> None:
        what = {"pending": "will RUN", "skipped": "is SKIPPED", "disabled": "is DISABLED"}.get(self.state, self.state)
        md(f"**12F — {self.title}** {what}{(': ' + self.reason) if self.reason else ''}")

    def _guard(self, what: str) -> bool:
        if not self.active:
            print(f"[12F] {what} skipped — stage {self.state}: {self.reason}")
            return False
        if self.state == "pending":
            self.state = "running"
            self.run.stage_begin(TRACK, self.title)
        return True

    def _fail(self, what: str, e: BaseException) -> None:
        self.state, self.reason = "failed", f"{type(e).__name__} in {what}: {e}"
        self.run.stage_fail(TRACK, e)
        print(f"\n{'=' * 100}\n[12F] FAILED in `{what}`: {type(e).__name__}: {e}\n{'=' * 100}")
        if not self.ucfg.continue_on_error:
            raise e

    def _call(self, what: str, fn: Any) -> Any:
        if not self._guard(what):
            return None
        try:
            return fn()
        except Exception as e:                                      # noqa: BLE001
            self._fail(what, e)
            return None

    def add_report_table(self, title: str, df: Any, cols: Any = None) -> None:
        if df is not None and getattr(df, "height", 0):
            self.report_tables.append((title, df, cols))

    # ================================================================== data
    def _bundle_dir(self) -> Path:
        if self.sources.bundle_dir is not None:
            return Path(self.sources.bundle_dir)
        from ..data.builder import build_bundle_from_repo

        out = Path(self.ucfg.bundle_dir) if self.ucfg.bundle_dir else Path(self.ucfg.drive_root) / "bundles" / "m12f_v1"
        self.run.log(f"building the bundle from the warehouse at {self.sources.research_root} → {out}")
        res = build_bundle_from_repo(repo_root=self.sources.research_root, output_dir=out, include_transfer=True, include_tcn=True, reuse_existing=True)
        print(res.text())
        self.sources.bundle_dir = out
        return out

    def open(self) -> Any:
        """Create (or reopen) the nflbowl experiment under ``RUN_DIR/checkpoints/experiments/<name>``. The science-hash suffix keeps
        incompatible science configurations in separate directories (their checkpoints can never mix)."""
        def go() -> Any:
            from .. import Experiment
            from ..config import load_config

            bundle = self._bundle_dir()
            u = self.ucfg
            ov = {**u.runtime_overrides(), **u.encoder_science_overrides}
            if u.encoder_experiment_dir:
                self.exp = Experiment.open(u.encoder_experiment_dir, data_dir=str(bundle), runtime={k.split(".", 1)[1]: v for k, v in u.runtime_overrides().items()})
            else:
                h = load_config(u.encoder_preset, ov).science_hash()
                self.exp = Experiment(data_dir=str(bundle), output_dir=str(self.run.root / "checkpoints"), config=u.encoder_preset, name=f"{u.experiment_name}_{h[:8]}", overrides=ov)
            self.run.save_model_metadata({"experiment_id": self.exp.id, "root": str(self.exp.root), "preset": u.encoder_preset, "science_hash": self.exp.cfg.science_hash(),
                                          "runtime": self.exp.cfg.to_dict()["runtime"], "science": self.exp.cfg.to_dict()["science"]}, "12f_experiment", TRACK)
            md(f"nflbowl experiment `{self.exp.id}` at `{self.exp.root}` · preset `{u.encoder_preset}` · science `{self.exp.cfg.science_hash()}` · precision `{self.exp.cfg.runtime.precision}`")
            return self.exp
        return self._call("open", go)

    def prepare(self) -> Any:
        """Stage the bundle to local disk, validate, derive per-sequence targets and F0 features (cached in the experiment)."""
        def go() -> Any:
            summary = self.exp.prepare(verify=self.ucfg.verify_bundle_sha256, reuse=True)
            self.run.save_processed_dataset(summary, "12f_bundle_summary", TRACK, {"bundle": summary.get("fingerprint")})
            display_df(self.exp.list_models())
            return summary
        return self._call("prepare", go)

    # ================================================================== training (separate from evaluation)
    def _train(self, model: str, **kw: Any) -> Any:
        t = self.exp.train(model, resume=self.ucfg.resume, **kw)
        self.trained.append({"model": model, **{k: str(v) for k, v in kw.items()}})
        display_df(t)
        from .colab import gpu_memory_summary

        print("GPU memory:", gpu_memory_summary())
        return t

    def train_primary(self) -> Any:
        """F4 transductive (primary). Resumes from the last epoch checkpoint after a disconnect; a finished run is restored, not retrained."""
        if not self.ucfg.run_gpu_training:
            md("_Training disabled (`run_gpu_training = False`): embeddings / evaluation use existing checkpoints only._")
            return None
        self.training_table = self._call("train_primary", lambda: self._train(self.ucfg.encoder_primary))
        return self.training_table

    def train_extras(self) -> Any:
        """Opt-in: other families and the ablations / negative controls (transductive)."""
        names = [*self.ucfg.encoder_families_extra]
        if self.ucfg.encoder_ablations:
            from ..models import registry as reg

            names += reg.ablations()
        if not names or not self.ucfg.run_gpu_training:
            md("_No extra encoders requested (encoder_families_extra / encoder_ablations)._")
            return None

        def go() -> Any:
            import polars as pl

            return pl.concat([self._train(n) for n in names], how="diagonal_relaxed")
        return self._call("train_extras", go)

    def train_inductive(self) -> Any:
        """Opt-in: F2 and F4 × outer folds (``science.inductive_configs``) — the only encoders that enter NFL prediction."""
        if not (self.ucfg.encoder_inductive and self.ucfg.run_gpu_training):
            md("_Inductive encoders not requested (`encoder_inductive = False`): no inductive retrieval, no transfer ladder, no 12G encoder block._")
            return None

        def go() -> Any:
            import polars as pl

            names = list(self.exp.cfg.science.inductive_configs)
            out = pl.concat([self._train(n, regime="inductive") for n in names], how="diagonal_relaxed")
            self.inductive_models = names
            return out
        return self._call("train_inductive", go)

    def train_learning_curve(self) -> Any:
        """Opt-in: transductive F4 with 10 % / 30 % of the NFL windows."""
        if not (self.ucfg.encoder_learning_curve and self.ucfg.run_gpu_training):
            md("_Learning curve not requested (`encoder_learning_curve = False`)._")
            return None

        def go() -> Any:
            import polars as pl

            return pl.concat([self._train(self.ucfg.encoder_primary, nfl_fraction=f) for f in self.exp.cfg.science.learning_curve_fractions], how="diagonal_relaxed")
        return self._call("train_learning_curve", go)

    def resume_unfinished(self) -> Any:
        """Continue every training job left `running` by a disconnect."""
        return self._call("resume", lambda: self.exp.resume())

    # ================================================================== embeddings + label-free evaluation
    def _trained_names(self) -> list[str]:
        from ..models import registry as reg

        have = {a["key"].split("__")[0] for a in self.exp.manifest.artifacts("training") if a["key"].endswith("__transductive")}
        return [reg.resolve(m) for m in [self.ucfg.encoder_primary, *self.ucfg.encoder_families_extra] if reg.resolve(m) in have] + \
               sorted(n for n in have if n not in {reg.resolve(m) for m in [self.ucfg.encoder_primary, *self.ucfg.encoder_families_extra]})

    def embed_and_evaluate(self) -> Any:
        """Embeddings (saved, reused) and label-free evaluation of the baselines and every trained transductive encoder; inductive
        held-out retrieval when inductive encoders exist. CPU-heavy evaluation runs after all GPU training."""
        def go() -> Any:
            import polars as pl

            from ..models import registry as reg

            lvl = self.ucfg.encoder_eval_level
            rows = []
            for m in [*self.ucfg.encoder_baselines, *self._trained_names()]:
                if reg.resolve(m) not in (reg.ENGINEERED, reg.RANDOM):
                    self.exp.embed(m, reuse=True)
                rows.append(self.exp.evaluate(m, level=lvl, reuse=True))
            prim = reg.resolve(self.ucfg.encoder_primary)
            for f in self.exp.cfg.science.learning_curve_fractions:              # learning-curve encoders, when trained
                if self.exp.manifest.find("training", f"{prim}__lc{int(round(f * 100))}") is not None:
                    self.exp.embed(prim, nfl_fraction=f, reuse=True)
                    rows.append(self.exp.evaluate(prim, level=lvl, nfl_fraction=f, reuse=True))
            ind_have ={a["key"].split("__")[0] for a in self.exp.manifest.artifacts("training") if "__inductive_f" in a["key"]}
            self.inductive_models = sorted(ind_have) or self.inductive_models
            ind = []
            if self.inductive_models:
                for m in self.inductive_models:
                    self.exp.embed(m, regime="inductive", reuse=True)
                    ind.append(self.exp.evaluate(m, regime="inductive", reuse=True))
                ind.append(self.exp.evaluate("F0", regime="inductive", reuse=True))
            self.inductive = pl.concat(ind, how="diagonal_relaxed") if ind else pl.DataFrame()
            self.metrics = self.exp.compare(level=lvl)
            display_df(self.metrics)
            if self.inductive.height:
                s = self.inductive.group_by("representation").agg(pl.col("auc").mean().alias("cross_auc_mean_over_folds"), pl.col("auc").min().alias("min_fold_auc"),
                                                                  pl.col("n_athletes").sum().alias("n_athletes"))
                display_df(s)
                self.add_report_table("INDUCTIVE held-out retrieval (separate table)", s)
            self.add_report_table(f"TRANSDUCTIVE label-free diagnostics ({lvl} level)", self.metrics)
            return self.metrics
        return self._call("embed_and_evaluate", go)

    def transfer(self) -> Any:
        """Inductive NFL-transfer ladder on the 12C frozen target (needs inductive encoders and athletes.parquet with nfl.q90_shared)."""
        if not self.ucfg.encoder_transfer:
            md("_Transfer ladder disabled (`encoder_transfer = False`)._")
            return None
        if not self.active:
            print(f"[12F] transfer skipped — stage {self.state}: {self.reason}")
            return None
        if not self.inductive_models:
            md("_Transfer ladder not run: needs inductive encoders (`encoder_inductive = True`)._")
            return None
        b = self.exp.bundle
        if b.athletes is None or "nfl.q90_shared" not in b.athletes.columns:
            md("_Transfer ladder not run: the bundle has no athletes.parquet with `nfl.q90_shared` (rebuild with include_transfer)._")
            return None

        def go() -> Any:
            self.transfer_table = self.exp.transfer(models=self.inductive_models, reuse=True)
            t = self.transfer_table.select("model", "reference", "n_athletes", "delta_rmse", "delta_lo", "delta_hi", "verdict") if self.transfer_table.height else self.transfer_table
            display_df(t)
            self.add_report_table("NFL transfer: M1 ladder on the 12C frozen target (INDUCTIVE)", t)
            return self.transfer_table
        return self._call("transfer", go)

    # ================================================================== persist into RUN_DIR
    def collect(self) -> None:
        """Figures (from the saved tables), evaluation views, embeddings and checkpoint metadata → RUN_DIR."""
        def go() -> None:
            import polars as pl

            exp, run = self.exp, self.run
            figs = exp.plot(save=True)
            for name in figs:
                for fmt in ("png", "svg", "pdf"):
                    run.copy_in(exp.store.figures / fmt / f"{name}.{fmt}", f"figures/{fmt}/12f_{name}.{fmt}", TRACK, kind="figure")
                for src in sorted((exp.store.figures / "source").glob(f"{name}__*.csv")):
                    run.copy_in(src, f"figures/data/12f_{src.name}", TRACK, kind="figure_data")
            try:
                from IPython.display import display

                for f in figs.values():
                    display(f)
            except ImportError:
                pass
            import matplotlib.pyplot as plt

            for f in figs.values():
                plt.close(f)
            for view in ("metrics", "probes", "identity_null", "inductive_retrieval", "transfer", "training_curves"):
                p = exp.store.evaluation / f"{view}.csv"
                if p.is_file():
                    run.save_table(pl.read_csv(p, infer_schema_length=None), f"12f_{view}", TRACK, note=f"view of {exp.id}")
            rp_ = exp.store.evaluation / "retrieval_results.parquet"
            if rp_.is_file():
                run.save_table(pl.read_parquet(rp_), "12f_retrieval_results", TRACK)
            idx = []
            for a in exp.manifest.artifacts("embeddings"):
                src = exp.manifest.resolve(a)
                dst = run.copy_in(src, f"embeddings/{Path(src).name}", TRACK, kind="embeddings") if self.ucfg.save_embeddings_copy else None
                idx.append({"key": a["key"], "source": str(src), "copy": run.rel(dst) if dst else None, "fold": a.get("fold"), "dims": a.get("dims"), "sha256": a.get("sha256")})
            if idx:
                run.save_table(pl.DataFrame(idx, infer_schema_length=None), "INDEX", TRACK, subdir="embeddings")
            for a in exp.manifest.artifacts("training"):
                run.save_model_metadata({"key": a["key"], "best_checkpoint": str(exp.manifest.resolve(a)), "sha256": a.get("sha256"), "deps": a.get("deps"),
                                         "checkpoint_dir": a.get("checkpoint_dir"), "signature": a.get("signature"), "fold": a.get("fold"), "curve": a.get("curve"),
                                         "experiment": exp.id}, f"12f_{a['key']}", TRACK)
            report = exp.report()
            run.copy_in(report, "reports/markdown/12f_nflbowl_experiment_report.md", TRACK, kind="report")
        self._call("collect", go)

    def inductive_byf(self, model: str | None = None) -> dict[int, Any] | None:
        """{fold: athlete embedding table} of the inductive primary encoder (for the 12G tournament), or None."""
        if self.exp is None or self.state == "failed":
            return None
        from ..models import registry as reg

        name = reg.resolve(model or self.ucfg.encoder_primary)
        if name not in self.inductive_models:
            return None
        try:
            return self.exp.inductive_athlete_embeddings(name)
        except Exception as e:                                      # noqa: BLE001
            self.run.log(f"inductive athlete embeddings unavailable: {e}", level="WARN")
            return None

    def publish_to_warehouse(self) -> None:
        """Write the 12F comparison / evidence tables, the inductive fold embeddings and the 12F run manifest into the warehouse store
        (exactly the names the 12F notebook wrote), so 12G's registry and tournament include 12F. No-op without a warehouse."""
        if self.ctx is None or self.metrics is None:
            md("_12F results not published to the warehouse (no warehouse context or no evaluation): 12G will list 12F as execution_pending._")
            return

        def go() -> None:
            import polars as pl
            from nflbdb.research.advanced_movement.m12 import advanced_evaluation as ae
            from nflbdb.research.advanced_movement.m12 import config as m12c
            from nflbdb.research.advanced_movement.m12 import reporting as rp
            from nflbdb.research.advanced_movement.m12 import shared_movement_encoder as se
            from nflbdb.research.advanced_movement.m12 import synthesis as sy

            ctx = self.ctx
            ctx.set_stage(TRACK)
            store, C, rm = ctx.store, ctx.C, ctx.controls.run_mode
            W = store.write_frame
            EVT = self.metrics
            TRF = self.transfer_table if self.transfer_table is not None else pl.DataFrame()
            for name in self.inductive_models:
                for f, t in (self.exp.inductive_athlete_embeddings(name) or {}).items():
                    W(t.with_columns(pl.lit(f).alias("held_out_fold")), f"m12f_inductive_{name}_f{f}")
            COMP = self.comparison_frame(rm)
            W(COMP, "m12f_comparisons")
            W(EVT, "m12f_eval_transductive")
            if self.inductive is not None and self.inductive.height:
                W(self.inductive, "m12f_eval_inductive")
            if TRF.height:
                W(TRF, "m12f_transfer")
            ev_rows = []
            for r in EVT.iter_rows(named=True):
                ev_rows.append(ae.evidence_row(f"12F.{r['representation']}.transductive", "12F", experiment="6", physical_interpretation="learned sequence representation (physics-probed)",
                                               input_data="sequence store (bundle)", output="retrieval auc", n_athletes=r.get("cross_n_athletes"), cross_domain_stat=r.get("cross_auc"),
                                               cross_domain_lo=r.get("cross_auc_lo"), cross_domain_hi=r.get("cross_auc_hi"), reliability=r.get("within_auc"),
                                               compute_cost="seconds" if str(r["representation"]).startswith(("F0", "random")) else "hours", training_regime=m12c.TRANSDUCTIVE,
                                               evaluation_target=m12c.PHYSICAL_TRANSFER, run_mode=rm, config_hash=C.hash(), limitations="transductive: label-free diagnostics only"))
            for r in (TRF.filter(pl.col("reference") == "M1").iter_rows(named=True) if TRF.height else []):
                ev_rows.append(ae.evidence_row(f"12F.{r['model']}.inductive", "12F", experiment="6", physical_interpretation="inductive fold embedding", input_data="sequence store (bundle)",
                                               output="nfl.q90_shared", n_athletes=r["n_athletes"], delta_rmse=r["delta_rmse"], delta_lo=r["delta_lo"], delta_hi=r["delta_hi"], compute_cost="hours",
                                               training_regime=m12c.INDUCTIVE, evaluation_target=m12c.PHYSICAL_TRANSFER, run_mode=rm, config_hash=C.hash()))
            W(ae.evidence_frame(ev_rows), "m12f_evidence")
            status = self._status(COMP)
            rp.run_manifest(store, "12F", ctx.provenance("12F", engine="nflbowl", experiment=self.exp.id, bundle=self.exp.bundle.fingerprint, science_hash=self.exp.cfg.science_hash()),
                            ["m12f_eval_transductive", "m12f_eval_inductive", "m12f_transfer", "m12f_comparisons", "m12f_evidence"], status,
                            {"device": self.exp.device, "inductive": bool(self.inductive_models and TRF.height)})
            self.m12_status = status
        self._call("publish_to_warehouse", go)

    def comparison_frame(self, run_mode: str | None = None) -> Any:
        """12F rows in the shared comparison schema (``synthesis.comparison_row``), exactly as the 12F notebook built them: one
        'AUC − 0.5' row per transductive representation, one Δ RMSE row per inductive ladder comparison (primary: M1+F4 vs M1 / M1+F0)."""
        import polars as pl
        from nflbdb.research.advanced_movement.m12 import config as m12c
        from nflbdb.research.advanced_movement.m12 import shared_movement_encoder as se
        from nflbdb.research.advanced_movement.m12 import synthesis as sy

        rm = run_mode or (self.ctx.controls.run_mode if self.ctx is not None else (self.exp.cfg.science.run_mode if self.exp is not None else "full"))
        EVT = self.metrics if self.metrics is not None else pl.DataFrame()
        TRF = self.transfer_table if self.transfer_table is not None else pl.DataFrame()
        comps = []
        for r in EVT.iter_rows(named=True):
            if r.get("cross_auc") is not None:
                lo, hi = r.get("cross_auc_lo"), r.get("cross_auc_hi")
                comps.append(sy.comparison_row("12F", f"retrieval AUC {r['representation']} − 0.5 (transductive)", "AUC − 0.5", r["cross_auc"] - 0.5,
                                               (lo - 0.5) if lo is not None else None, (hi - 0.5) if hi is not None else None, r.get("cross_n_athletes"), m12c.PHYSICAL_TRANSFER,
                                               "m12f_eval_transductive", rm, experiment=6, training_regime=m12c.TRANSDUCTIVE))
        for r in (TRF.iter_rows(named=True) if TRF.height else []):
            comps.append(sy.comparison_row("12F", f"{r['model']} vs {r['reference']} → nfl.q90_shared", "Δ RMSE", r["delta_rmse"], r["delta_lo"], r["delta_hi"], r["n_athletes"], m12c.PHYSICAL_TRANSFER,
                                           "m12f_transfer", rm, experiment=6, training_regime=m12c.INDUCTIVE,
                                           primary=(r["model"], r["reference"]) in ((f"M1+{se.PRIMARY}", "M1"), (f"M1+{se.PRIMARY}", "M1+F0"))))
        return sy.comparison_frame(comps)

    def _status(self, comp: Any = None) -> str:
        import polars as pl
        from nflbdb.research.advanced_movement.m12 import reporting as rp

        prim = comp.filter(pl.col("primary"))["verdict"].to_list() if comp is not None and comp.height else []
        complete = bool(self.inductive_models and self.transfer_table is not None and self.transfer_table.height)
        return rp.status_from(prim, completed=complete)

    # ================================================================== conclusion / report
    def conclude(self) -> None:
        def go() -> None:
            import polars as pl
            from nflbdb.research.advanced_movement.m12 import reporting as rp

            EVT = self.metrics if self.metrics is not None else pl.DataFrame()
            TRF = self.transfer_table if self.transfer_table is not None else pl.DataFrame()
            ind = self.inductive if self.inductive is not None else pl.DataFrame()
            isum = (ind.group_by("representation").agg(pl.col("auc").mean().alias("m")) if ind.height else pl.DataFrame())
            b = self.exp.bundle.summary()
            comp = self.comparison_frame()
            self.run.save_table(comp, "m12f_comparisons", TRACK, note="12F comparison rows (shared schema)")
            status = self.m12_status or self._status(comp)
            self.m12_status = status
            self.facts = {"status": status, "run_mode": self.exp.cfg.science.run_mode,
                          "sample": f"{b['n_sequences']:,} sequences ({b['n_combine']:,} Combine, {b['n_nfl']:,} NFL), {b['n_athletes']} athletes; device {self.exp.device}; "
                                    f"precision {self.exp.cfg.runtime.precision}; preset {self.ucfg.encoder_preset}.",
                          "evidence": [f"{r['representation']} (transductive, {self.ucfg.encoder_eval_level}): cross AUC {rp.ci(r.get('cross_auc'), r.get('cross_auc_lo'), r.get('cross_auc_hi'))}, "
                                       f"domain AUC {rp.fmt(r.get('domain_auc'))}, probe R² median {rp.fmt(r.get('probe_r2_median'))}, identity-null mean {rp.fmt(r.get('null_mean'))}" for r in EVT.iter_rows(named=True)]
                                      + [f"{r['representation']} (inductive): mean held-out AUC {rp.fmt(r['m'])}" for r in isum.iter_rows(named=True)]
                                      + [f"{r['model']} vs {r['reference']}: Δ RMSE {rp.ci(r['delta_rmse'], r['delta_lo'], r['delta_hi'], 4)} — {r['verdict']}" for r in TRF.iter_rows(named=True)],
                          "alternatives": ["Retrieval can rise through position-typical or body-size-typical movement (retrieval is within position to limit this).",
                                           "Matched-state alignment may pull domains together without athlete information (compare F4 with F4_noalign and the random encoder)."],
                          "relevance": "Tests whether large unlabeled NFL movement improves a pre-draft movement representation beyond engineered physics.",
                          "limitations": ["One architecture size per family; no search.", "Wall-clock caps can stop training early (recorded per checkpoint).", "One cohort."]
                                         + ([] if self.inductive_models else ["No inductive encoder in this run: only label-free transductive diagnostics; the primary inductive hypothesis is untested."])
                                         + (["DEBUG preset (smoke budgets): numbers are wiring checks, not results."] if self.ucfg.encoder_preset == "fast_iteration" else []),
                          "next": ["Continue only if the inductive F4 beats F0 and random on held-out retrieval AND adds NFL prediction."]}
            self.conclusion_md = rp.conclusion_markdown("12F", self.facts)
            md(self.conclusion_md)
        self._call("conclude", go)

    def finish(self) -> Path | None:
        from .reporting import stage_report

        outcome ={"running": "executed", "failed": "failed"}.get(self.state, self.state)
        if outcome == "executed" and self.conclusion_md is None:
            outcome = "incomplete"
        if self.state in ("skipped", "disabled"):
            self.run.stage_begin(TRACK, self.state)
        paths = self.run.save_markdown(stage_report(self, outcome), self.report_name, TRACK)
        if self.state == "running":
            self.run.stage_end(TRACK, STATUS_DONE if outcome == "executed" else STATUS_FAILED, note=f"m12 status {self.m12_status}")
        elif self.state in ("skipped", "disabled"):
            self.run.stage_end(TRACK, STATUS_SKIPPED, note=self.reason)
        self.final = self.state
        self.state = "finished"
        if paths:
            md(f"Report: `{self.run.rel(paths[0])}`")
        return paths[0] if paths else None


__all__ = ["EncoderSection"]
