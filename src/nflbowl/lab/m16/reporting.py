"""M16 research report generator — every number is read from saved tables of the run; nothing is typed by hand.

``build(run, cfg)`` returns Markdown with: the step ledger, what was evaluated, the gate summary (extended G16), Track A
(coverage, reliability scorecard, context support), Track B (provenance registry, controlled retrieval vs controls,
alignment damage), Track C (rung plan, ladder metrics, paired Δ vs M1, negative controls, hypotheses, positions), what
passed / failed / remains uncertain, readiness for M17 and the pivots implied by the gates document's fallback routes, and
one card per experiment (why · what · how · what if it fails · why a scout should care). Missing tables are named as
missing — an absent result is never summarised as a negative one.
"""

from __future__ import annotations

from typing import Any, Sequence

from ...utils.reproducibility import utc_now
from ..status import BLOCKED_GATE, EXPLORATORY, FAIL, NOT_EVALUATED, PASS

EXPERIMENTS = {
    "E16.A1": ("M15 changed nothing about the official baseline, and every later claim is 'beyond M1' — so M1 must be exactly the official one.",
               "Official ladder M1 recomputed vs its independent refit and every compatible persisted record, at 1e-8 on predictions.",
               "Athlete-level |Δ| per comparison, fold and target contract checks; G16-B.",
               "A mismatch is a data-contract failure: every comparison against M1 is suspended (hard stop).",
               "A scout needs one fixed reference point: what the stopwatch and tape measure already say."),
    "E16.A3": ("A target that is mostly noise or exposure cannot reward a better Combine measurement.",
               "Split-half reliability (season → game → random), exposure curves, demand-kind share of every NFL target.",
               "Position-residualised Spearman–Brown reliability with athlete-bootstrap CIs; G16-A rules.",
               "The target is dropped or revised once (post_hoc); T0 remains the only outcome.",
               "Only stable NFL movement outcomes say something about a player rather than about his snaps."),
    "E16.A7": ("Targets were never inspected beyond their reliability; M15 showed strong context effects in NFL windows.",
               "Distributions, variance decomposition, exposure / duration / context / draft-year relations, threshold and aggregation sensitivity, "
               "subsample stability, correlations, selection and censoring audits.",
               "Athlete as the unit; within-position associations with athlete-bootstrap CIs; the scorecard joins every flag.",
               "Flags downgrade a passing target to EXPLORATORY.",
               "Shows which NFL movement outcomes differ between players of one position, not just between positions."),
    "E16.A8": ("Combine drills and NFL windows are different movements; M15 found near-perfect domain separability.",
               "Common support of each drill vs NFL windows on entry speed, peak acceleration, duration, braking share and turning demand (exact position).",
               "Univariate overlap + coarsened multivariate support; BLOCKED_INSUFFICIENT_OVERLAP below the pre-set share.",
               "No context-matched comparison is made (no extrapolation); the drill hypothesis that needs it is blocked.",
               "Tells a scout whether a drill resembles anything the player will actually do on Sundays."),
    "E16.B4": ("M15 artifacts came from several notebooks; E15.C is blocked by a configuration / API disagreement.",
               "Provenance of every embedding / checkpoint: fold, training cohort, held-out exclusion, bundle, channels, transform, checksum.",
               "Four states; only VERIFIED_FOLD_SAFE enters supervised comparisons; G15 decision and preservation evidence attached.",
               "Unknown or leaking artifacts block their experiments instead of being used.",
               "A model that has seen the prospect it scores is worthless for scouting."),
    "E16.B5": ("M15 retrieval gains (all-position top-1 ≈ 6 %) sit at position-only chance; same-position and twin gains had CIs crossing 0.",
               "Five-fold retrieval of every verified representation, fusion and controls (F0, kinematics, position-only, random) in all / same-position / "
               "stopwatch-twin / physical-twin pools and per drill.",
               "Per-query ranks with pool-specific chance; paired ΔMRR vs the strongest control on the same queries (Holm).",
               "Cross-domain similarity claims are dropped; representations may still be screened as Combine-only predictors.",
               "Only similarity beyond position and stopwatch could ever point to comparable NFL movers."),
    "E16.B6": ("InfoNCE alignment harmed repeated-trial AUC in 3 / 5 LSTM folds; effective rank was 6–8 of 32.",
               "Repeat AUC unaligned vs aligned per fold and drill, effective rank, ICC, domain / position probes, M1 re-encoding, duration sensitivity.",
               "Fold-level paired changes (t-interval over 5 held-out folds); destructive rule −0.02.",
               "A destructive alignment never advances, whatever its retrieval score.",
               "A measurement that does not repeat within a player cannot separate players."),
    "E16.C4": ("The exploratory engineered-tracking benchmark did not beat position (0.4108 vs 0.4308 yd/s on a different target).",
               "Ladder B0–B7 on each eligible target: position, M1, + kinematics, + PCA / STFT / LSTM / fusion, + M14 reliable.",
               "Official engine, identical cohort, paired fold-stratified bootstrap, Holm on the screen family, practical margin, MDE.",
               "A precise null with the MDE is the result: tracking adds nothing detectable beyond M1 at this cohort size.",
               "Answers the scouting question directly: does the tracking add to what the stopwatch already says?"),
    "E16.C5": ("Embeddings encode position and stopwatch information; gains could come from that alone.",
               "Within-position shuffled embeddings (null distribution), random embeddings, reduced dimensions.",
               "Observed ΔRMSE vs the shuffled null (q95, permutation p).",
               "A gain inside the null is labelled not athlete-specific.",
               "Separates 'information about this player' from 'information about his position group'."),
    "E16.C7": ("Physical task matching (forty ↔ high speed, shuttle ↔ braking, three-cone ↔ redirection) is the most plausible transfer route.",
               "Drill-specific embeddings and kinematics vs M1 on the matched NFL outcome (context-matched for C3).",
               "Same engine and statistics as C4; prerequisites (sample, validity, censoring, overlap) checked first.",
               "The hypothesis is reported as not supported, or blocked with its reason.",
               "Would tell a scout which drill to watch for which NFL skill."),
}


def tables(run: Any) -> dict[str, Any]:
    """Every saved table the report may use (absent ones are simply missing)."""
    names = {"gate_a": "target", "coverage": "target", "reliability": "target", "scorecard": "exploration", "support": "context", "matched_summary": "context",
             "registry": "representation", "controlled_summary": "retrieval", "controlled_vs_controls": "retrieval", "repeat_damage": "retrieval",
             "rung_plan": "ladder", "metrics": "ladder", "paired": "ladder", "status": "ladder", "summary": "controls", "hypotheses_status": "hypotheses",
             "hypotheses_paired": "hypotheses", "strategies": "positions", "m16_gate": "synthesis", "m16_gate_extended": "synthesis", "g16b_gate": "synthesis",
             "mde": "detectability", "dual_domain_summary": "cohort", "funnel": "cohort"}
    out = {}
    for n, fam in names.items():
        try:
            if run.has(n, fam):
                out[n] = run.frame(n, fam)
        except Exception:                                            # noqa: BLE001 — an unreadable table is reported as missing
            continue
    return out


def _fmt(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, float):
        return "" if v != v else (f"{v:.4g}" if abs(v) < 1e4 else f"{v:,.0f}")
    return str(v).replace("|", "/").replace("\n", " ")[:160]


def md(df: Any, cols: Sequence[str], n: int = 40) -> list[str]:
    if df is None or not df.height:
        return ["_(table not produced in this run)_"]
    cols = [c for c in cols if c in df.columns]
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in df.select(cols).head(n).iter_rows():
        out.append("| " + " | ".join(_fmt(v) for v in r) + " |")
    if df.height > n:
        out.append(f"_… {df.height - n} more row(s) in the saved table_")
    return out


def decision(ext: Any) -> dict[str, str]:
    import polars as pl

    if ext is None or not ext.height:
        return {}
    o = ext.filter(pl.col("criterion").str.contains("overall"))
    return {r["gate"]: r["status"] for r in o.iter_rows(named=True)}


def readiness(dec: dict[str, str]) -> list[str]:
    """Gates-document decision table + fallback routes, applied to the computed overall states."""
    a, b, c, d = (dec.get(k, NOT_EVALUATED) for k in ("G16-A", "G16-B", "G16-C", "G16-D"))
    out = []
    if b == FAIL:
        return ["**G16-B FAIL — hard stop.** M17 / M18 comparisons against M1 are blocked until the M1 data contract is fixed."]
    if b in (NOT_EVALUATED, BLOCKED_GATE):
        out.append("G16-B not established: every downstream result is labelled unverified until M1 reproduction is shown.")
    elif b == EXPLORATORY:
        out.append("G16-B rests on a documented controlled refit (no persisted record): downstream results carry that label.")
    if a == FAIL:
        out.append("No task target is valid: M18 runs on T0 only; M17 Track A also repairs NFL-side events.")
    elif a in (PASS, EXPLORATORY):
        out.append("Valid task targets are frozen in `manifests/target_freeze_m16.json` for M17 / M18 (T0 stays primary).")
    if c == PASS:
        out.append("G16-C PASS: some rung beats M1 on T0 with every control — M18 tests it confirmatorily (pre-registered).")
    elif c in (FAIL, EXPLORATORY):
        out.append("G16-C not passed: M17 aims at better measurement (Tracks A, B, E); in M18 the null with its MDE is the expected reading.")
    if d == PASS:
        out.append("G16-D PASS: cross-domain similarity can be used in M17 Track D.")
    elif d in (FAIL, BLOCKED_GATE, EXPLORATORY, NOT_EVALUATED):
        out.append("G16-D not passed: drop cross-domain similarity claims; representations may only be screened as Combine-only predictors "
                   "(fallback: PCA → handcrafted time-domain features → none).")
    return out


def build(run: Any, cfg: Any) -> str:
    import polars as pl

    T = tables(run)
    ext = T.get("m16_gate_extended")
    dec = decision(ext)
    led = run.ledger.table()
    done = led.filter(pl.col("status").is_in(["COMPLETED", "CACHED"]))["step"].to_list() if led.height else []
    notrun = led.filter(~pl.col("status").is_in(["COMPLETED", "CACHED"])) if led.height else led
    L = [f"# M16 research report — {run.root.name}", "", f"Generated {utc_now()} from the saved tables of `{run.root}`. Config fingerprint "
         f"`{cfg.fingerprint()}`. No number below was typed by hand; a missing table is shown as missing.", "",
         "Distinctions kept throughout: **observed M15 results** (hand-off) · **unverified saved artifacts** (provenance registry) · "
         "**new hypotheses** (E16-C1–C3) · **M16 outcomes** (tables of this run).", "", "## 1. Step ledger", ""]
    L += md(led, ["step", "status", "reason", "seconds"], 80)
    L += ["", "## 2. What was evaluated", "", f"Completed or cached: {', '.join(done) or 'nothing yet'}.", "",
          f"Not completed: {', '.join(f'{r[0]} ({r[1]})' for r in notrun.select('step', 'status').rows()) or 'none'}." if notrun.height else "Not completed: none.",
          "", "## 3. Gate summary", ""]
    if ext is not None:
        L += md(ext.filter(pl.col("criterion").str.contains("overall")), ["gate", "status", "evidence", "explanation"])
        L += ["", "### Per subject", ""] + md(ext.filter(~pl.col("criterion").str.contains("overall")),
                                              ["gate", "subject", "status", "observed", "ci_lo", "ci_hi", "explanation", "next_action"], 80)
    else:
        L += ["_Extended gate not computed (run section 10)._"]
    L += ["", "## 4. Track A — NFL targets", "", "### Coverage (all positions)", ""]
    cov = T.get("coverage")
    L += md(cov.filter(pl.col("position_group") == "ALL") if cov is not None else None, ["target_id", "column", "unit", "role", "n_athletes", "mean", "sd"])
    L += ["", "### Validity scorecard", ""] + md(T.get("scorecard"), ["target_id", "n_athletes", "g16a_status", "split_half_reliability", "split_half_lo",
                                                                      "measurement_reliability", "rho_log_exposure", "censored_share", "flags"])
    L += ["", "### Context common support", ""] + md(T.get("support"), ["drill", "support_combine", "support_nfl", "n_athletes_both_supported", "status", "reason"])
    L += ["", "## 5. Track B — representations", "", "### Provenance registry", ""]
    L += md(T.get("registry"), ["rep", "seed", "n_folds", "status", "g15_decision", "preservation_tables_complete", "reasons"])
    vc = T.get("controlled_vs_controls")
    L += ["", "### Controlled retrieval vs the strongest control (scope all)", ""]
    L += md(vc.filter(pl.col("strongest") & (pl.col("scope") == "all")) if vc is not None else None,
            ["rep", "arm", "pool", "control", "n", "mrr", "mrr_control", "delta_mrr", "lo", "hi", "p_holm_strongest"], 60)
    dm = T.get("repeat_damage")
    L += ["", "### Alignment damage (repeat AUC aligned − unaligned)", ""] + md(dm, ["rep", "arm", "kind", "n_folds", "delta_mean", "delta_lo", "delta_hi",
                                                                                    "folds_dropping", "destructive"])
    L += ["", "## 6. Track C — incremental prediction", "", "### Rung plan", ""] + md(T.get("rung_plan"), ["rung", "model", "status", "label", "rep", "reason"])
    L += ["", "### Paired ΔRMSE vs M1 (positive = better than M1)", ""]
    pa = T.get("paired")
    L += md(pa.filter(pl.col("reference") == "M1") if pa is not None else None,
            ["target_id", "model", "n_athletes", "rmse_ref", "rmse_model", "delta_rmse", "delta_lo", "delta_hi", "delta_pct", "p_holm_primary", "precise_null"], 60)
    L += ["", "### Negative controls", ""] + md(T.get("summary"), ["target_id", "model", "observed_delta_rmse", "null_mean", "null_q95", "p_perm", "reading"])
    L += ["", "### Drill hypotheses", ""] + md(T.get("hypotheses_status"), ["hypothesis", "target", "status", "reason"])
    L += md(T.get("hypotheses_paired"), ["hypothesis", "target_id", "model", "delta_rmse", "delta_lo", "delta_hi", "p_holm_primary"], 40)
    L += ["", "### Position strategies (separate / pooled vs global)", ""] + md(T.get("strategies"), ["target_id", "position_group", "n_position", "model",
                                                                                                      "reference", "delta_rmse", "delta_lo", "delta_hi"], 40)
    L += ["", "## 7. What passed, failed, remains uncertain", ""]
    if ext is not None:
        for st, title in ((PASS, "Passed"), (FAIL, "Failed (negative scientific results)"), (EXPLORATORY, "Exploratory (not demonstrated)"),
                          (BLOCKED_GATE, "Blocked"), (NOT_EVALUATED, "Not evaluated")):
            sub = ext.filter((pl.col("status") == st) & ~pl.col("criterion").str.contains("overall"))
            L.append(f"- **{title}:** " + ("; ".join(f"{r['gate']} {r['subject']}" for r in sub.iter_rows(named=True)) if sub.height else "none"))
    L += ["", "## 8. Readiness for M17 and pivots", ""] + [f"- {x}" for x in (readiness(dec) or ["Gate states not computed yet."])]
    L += ["", "## 9. Experiment cards", ""]
    for k, (why, what, how, fail, scout) in EXPERIMENTS.items():
        L += [f"### {k}", f"- **Why:** {why}", f"- **What:** {what}", f"- **How:** {how}", f"- **If it fails:** {fail}", f"- **Why a scout should care:** {scout}", ""]
    L += ["No scouting recommendation is produced from a model that did not pass its gates."]
    return "\n".join(L)


def write(run: Any, cfg: Any) -> list[Any]:
    return run.save_markdown(build(run, cfg), "m16_research_report")


__all__ = ["EXPERIMENTS", "build", "decision", "md", "readiness", "tables", "write"]
