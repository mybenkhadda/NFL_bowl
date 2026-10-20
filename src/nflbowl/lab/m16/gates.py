"""G16 — verified baselines and cross-domain validity, from computed M16 tables only.

    G16-A  target reliability       PASS when ≥ 1 task target passes ``target_reliability.gate_a`` (T0 stays primary unless the
                                    analyst records otherwise); FAIL when every computed task target fails; NOT_EVALUATED otherwise
    G16-B  M1 reproducibility       the overall row of ``g16b_gate`` (M1 verification)
    G16-C  incremental information  on the primary target: PASS if M1+simple or M1+M14_reliable is ``validated`` vs M1;
                                    PRECISE_NULL if both comparisons satisfy the precise-null rule; INCONCLUSIVE otherwise
    G16-D  cross-domain similarity  PASS when some eligible representation has (same-position MRR vs F0 CI > 0, Holm p ≤ 0.05)
                                    AND (all-pool top-1 above position-only, CI > 0) AND a CEM-matched same-position MRR not below
                                    the original by more than half the F0 margin; FAIL when tests exist and none passes
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..status import BLOCKED_GATE, EXPLORATORY, FAIL, NOT_EVALUATED, PASS


def g16(gate_a: Any, g16b: Any, status_primary: Any, paired_primary: Any, primary_tests: Any, cem: Any) -> Any:
    import polars as pl

    rows = []
    # A
    if gate_a is None or not gate_a.height:
        rows.append({"criterion": "G16-A target reliability", "status": NOT_EVALUATED, "note": "no reliability table"})
    else:
        task = gate_a.filter(pl.col("target_id") != "T0")
        st = task["status"].to_list()
        s = PASS if PASS in st else (FAIL if st and all(x == FAIL for x in st) else NOT_EVALUATED)
        rows.append({"criterion": "G16-A target reliability", "status": s, "note": f"passing task targets: {task.filter(pl.col('status') == PASS)['target_id'].to_list()}"})
    # B
    if g16b is None or not g16b.height:
        rows.append({"criterion": "G16-B M1 reproducibility", "status": NOT_EVALUATED, "note": "no M1 verification"})
    else:
        o = g16b.filter(pl.col("criterion") == "G16-B overall")
        rows.append({"criterion": "G16-B M1 reproducibility", "status": o["status"][0] if o.height else NOT_EVALUATED, "note": o["note"][0] if o.height else ""})
    # C
    if status_primary is None or not status_primary.height:
        rows.append({"criterion": "G16-C incremental information", "status": NOT_EVALUATED, "note": "no baseline ladder on the primary target"})
    else:
        sel = status_primary.filter(pl.col("model").is_in(["M1+simple", "M1+M14_reliable"]) & (pl.col("reference") == "M1"))
        pn = paired_primary.filter(pl.col("model").is_in(["M1+simple", "M1+M14_reliable"]) & (pl.col("reference") == "M1")) if paired_primary is not None else None
        if (sel["status"] == "validated").any():
            s = PASS
        elif pn is not None and pn.height and "precise_null" in pn.columns and bool(pn["precise_null"].all()):
            s = "PRECISE_NULL"
        elif sel.height:
            s = "INCONCLUSIVE"
        else:
            s = NOT_EVALUATED
        rows.append({"criterion": "G16-C incremental information", "status": s, "note": "; ".join(f"{r['model']}: {r['status']}" for r in sel.iter_rows(named=True))})
    # D
    if primary_tests is None or not primary_tests.height:
        rows.append({"criterion": "G16-D cross-domain similarity", "status": NOT_EVALUATED, "note": "no retrieval tests (no eligible representation?)"})
    else:
        ok_reps = []
        for rep in primary_tests["rep"].unique().to_list():
            t = primary_tests.filter(pl.col("rep") == rep)
            m = t.filter(pl.col("test").str.starts_with("same_position"))
            pchk = t.filter(pl.col("test").str.starts_with("all-pool"))
            a_ok = bool(m.height and m["lo"][0] > 0 and (m["p_holm"][0] is not None and m["p_holm"][0] <= 0.05))
            b_ok = bool(pchk.height and pchk["lo"][0] > 0)
            c_ok = True
            if cem is not None and cem.height:
                c = cem.filter(pl.col("rep") == rep).group_by("condition").agg(pl.col("same_position_mrr").mean())
                o = c.filter(pl.col("condition") == "original")["same_position_mrr"]
                mt = c.filter(pl.col("condition") == "cem_matched")["same_position_mrr"]
                if o.len() and mt.len() and m.height:
                    c_ok = bool(np.isfinite(mt[0]) and (o[0] - mt[0]) <= 0.5 * float(m["estimate"][0]))
            if a_ok and b_ok and c_ok:
                ok_reps.append(rep)
        rows.append({"criterion": "G16-D cross-domain similarity", "status": PASS if ok_reps else FAIL, "note": f"passing: {ok_reps}"})
    return pl.DataFrame(rows, infer_schema_length=None)


# ====================================================================== extended, evidence-rich gate (M16 laboratory)
# ``g16`` above stays the table M17 / M18 read (``tables/synthesis/m16_gate``). ``g16_extended`` writes
# ``tables/synthesis/m16_gate_extended``: one row per gate × subject with the evidence, the rule, the observed value, its CI,
# the artifact, an explanation and the next action. States: PASS · FAIL · EXPLORATORY · BLOCKED · NOT_EVALUATED.
# Naming follows the gates document (G16-C = incremental information, G16-D = cross-domain similarity); the M16 laboratory
# brief calls them "incremental prediction" and "transfer / representation validity" — same content, document IDs kept.
# Pre-specified additions (documented in M16_EXPERIMENTS_AND_VALIDATION.md, fixed before execution):
#   G16-C rung PASS additionally needs Δ ≥ practical margin, a gain above the within-position shuffled null, verified
#   provenance and (for representations) G15 GO; G16-D additionally needs the stopwatch-twin pool vs the strongest control
#   (CI > 0) and a non-destructive alignment when an aligned arm is the one passing.
NEXT = {PASS: "eligible for M17 / M18 as specified", FAIL: "record as a negative result; use the fallback route of the gates document",
        EXPLORATORY: "report descriptively; needs the missing evidence (G15 GO / persisted record / control) before any claim",
        BLOCKED_GATE: "resolve the blocking input named in the explanation and rerun the step", NOT_EVALUATED: "run the producing step and send back its tables"}


def _x(gate: str, criterion: str, subject: str, status: str, evidence: str = "", rule: str = "", observed: Any = None, lo: Any = None, hi: Any = None,
       artifact: str = "", explanation: str = "", next_action: str | None = None) -> dict[str, Any]:
    def f(v: Any) -> Any:
        try:
            return None if v is None or not np.isfinite(float(v)) else float(v)
        except (TypeError, ValueError):
            return None
    return {"gate": gate, "criterion": criterion, "subject": subject, "status": status, "evidence": evidence, "rule": rule, "observed": f(observed),
            "ci_lo": f(lo), "ci_hi": f(hi), "artifact": artifact, "explanation": explanation, "next_action": next_action or NEXT.get(status, "")}


def _overall(rows: list[dict[str, Any]], gate: str, any_pass: bool = True) -> dict[str, Any]:
    st = [r["status"] for r in rows if r["gate"] == gate]
    if not st:
        s = NOT_EVALUATED
    elif any_pass and PASS in st:
        s = PASS
    elif not any_pass and st and all(x == PASS for x in st):
        s = PASS
    elif EXPLORATORY in st:
        s = EXPLORATORY
    elif FAIL in st and not any(x in (NOT_EVALUATED, BLOCKED_GATE) for x in st):
        s = FAIL
    elif all(x == BLOCKED_GATE for x in st):
        s = BLOCKED_GATE
    elif FAIL in st:
        s = FAIL
    else:
        s = NOT_EVALUATED
    subj = [r["subject"] for r in rows if r["gate"] == gate and r["status"] == PASS]
    return _x(gate, f"{gate} overall", "all", s, evidence=f"{len(st)} subject row(s); passing: {subj or 'none'}",
              rule="any subject PASS" if any_pass else "every criterion PASS", explanation="summary of the rows above")


def g16_extended(t: dict[str, Any], cfg: Any) -> Any:
    """``t``: saved tables by name (missing ones absent or None) — gate_a, scorecard, g16b, cohort_audit, plan, status, paired,
    controls_summary, registry, vs_controls, damage, cem, mde, legacy."""
    import polars as pl

    def g(name: str) -> Any:
        v = t.get(name)
        return v if v is not None and getattr(v, "height", 0) else None

    rows: list[dict[str, Any]] = []
    # ---------------- G16-A outcome validity
    ga, sc = g("gate_a"), g("scorecard")
    if ga is None:
        rows.append(_x("G16-A", "outcome validity", "all targets", NOT_EVALUATED, explanation="no tables/target/gate_a (run E16.A3)"))
    else:
        for r in ga.iter_rows(named=True):
            tid = r["target_id"]
            flags = sc.filter(pl.col("target_id") == tid)["flags"].to_list()[0] if sc is not None and sc.filter(pl.col("target_id") == tid).height else "not computed"
            st = r["status"]
            if st == PASS and flags not in ("none",):
                st, why = EXPLORATORY, f"G16-A rule passed but the target laboratory flags: {flags}"
            elif st == PASS:
                why = "reliability, coverage, exposure and context rules pass; no laboratory flag"
            elif tid == "T0":
                why = "12C frozen target: reliability lives upstream; it stays primary unless a task target is pre-declared instead"
            else:
                why = r.get("reason") or ""
            rows.append(_x("G16-A", "outcome validity", tid, st, evidence=f"reliability {r.get('reliability')}, ρ(exposure) {r.get('rho_exposure')}, "
                           f"context share {r.get('context_share')}; flags: {flags}", rule="gates document G16-A + no censoring / selection flag",
                           observed=r.get("reliability"), lo=r.get("rel_lo"), artifact="tables/target/gate_a · tables/exploration/scorecard", explanation=why,
                           next_action=None if st != FAIL else "revise the definition once before M18 (label post_hoc) or drop the target"))
    rows.append(_overall(rows, "G16-A"))
    # ---------------- G16-B baseline integrity
    gb, ca = g("g16b"), g("cohort_audit")
    if gb is None:
        rows.append(_x("G16-B", "M1 reproduction", "M1", NOT_EVALUATED, explanation="no tables/synthesis/g16b_gate (run E16.A1)"))
    else:
        o = gb.filter(pl.col("criterion") == "G16-B overall")
        st = o["status"][0] if o.height else NOT_EVALUATED
        internal = gb.filter(pl.col("criterion").str.starts_with("M1 internal"))
        if st == NOT_EVALUATED and internal.height and internal["status"][0] == PASS:
            st, why = EXPLORATORY, "internal reproduction at tolerance passed; no compatible persisted official record → documented controlled refit only"
        else:
            why = o["note"][0] if o.height else ""
        rows.append(_x("G16-B", "M1 reproduction (1e-8)", "M1", st, evidence="; ".join(f"{r['criterion']}: {r['status']}" for r in gb.iter_rows(named=True)),
                       rule="gates document G16-B (hard stop on FAIL)", observed=internal["value"][0] if internal.height else None,
                       artifact="tables/synthesis/g16b_gate · tables/baselines/m1_reproducibility", explanation=why,
                       next_action="HARD STOP: fix the data contract before any comparison" if st == FAIL else None))
    if ca is not None:
        for (tid,), c in ca.group_by("target_id"):
            ok = bool(c["identical_to_strict"].all())
            rows.append(_x("G16-B", "identical evaluation cohort across ladder models", str(tid), PASS if ok else FAIL,
                           evidence=f"strict cohort {int(c['n_strict_cohort'][0])} athletes; models outside it: "
                                    f"{c.filter(~pl.col('identical_to_strict'))['model'].to_list()}", rule="every compared model scored on the same athletes",
                           observed=int(c["n_strict_cohort"][0]), artifact="tables/ladder/cohort_audit",
                           explanation="paired comparisons are restricted to the strict cohort" if ok else "coverage differs: see the coverage analysis"))
    rows.append(_overall(rows, "G16-B", any_pass=False))
    # ---------------- G16-C incremental information (per rung, per target; decision on the primary target T0)
    plan, st_, pa, ctl = g("plan"), g("status"), g("paired"), g("controls_summary")
    margin = getattr(cfg, "practical_margin_pct", 2.0)
    if plan is None:
        rows.append(_x("G16-C", "incremental information", "all rungs", NOT_EVALUATED, explanation="no ladder plan (run E16.C4)"))
    else:
        targets = sorted(set(pa["target_id"].to_list())) if pa is not None and "target_id" in pa.columns else ["T0"]
        for tid in targets:
            for r in plan.filter(~pl.col("rung").is_in(["B0", "B1"])).iter_rows(named=True):
                subj = f"{tid} · {r['rung']} {r['model']}"
                if r["status"] != "eligible":
                    rows.append(_x("G16-C", "incremental information", subj, BLOCKED_GATE, evidence=r["reason"], rule="rung must be fittable from verified inputs",
                                   artifact="tables/ladder/rung_plan", explanation=f"{r['status']}: {r['reason']}"))
                    continue
                p = pa.filter((pl.col("target_id") == tid) & (pl.col("model") == r["model"]) & (pl.col("reference") == "M1")) if pa is not None else None
                s = st_.filter((pl.col("target_id") == tid) & (pl.col("model") == r["model"]) & (pl.col("reference") == "M1")) if st_ is not None else None
                if p is None or not p.height:
                    rows.append(_x("G16-C", "incremental information", subj, NOT_EVALUATED, explanation="no paired comparison for this rung / target"))
                    continue
                pr = p.row(0, named=True)
                declared = s["status"][0] if s is not None and s.height else "inconclusive"
                nul = ctl.filter((pl.col("target_id") == tid) & (pl.col("model") == r["model"])) if ctl is not None else None
                above = bool(nul["above_shuffled_null"][0]) if nul is not None and nul.height else None
                big = (pr.get("delta_pct") or -1e9) >= margin
                pn = bool(pr.get("precise_null")) if pr.get("precise_null") is not None else False
                emb = r["rung"] in ("B3", "B4", "B5", "B6")
                if declared == "validated" and big and (not emb or above) and r["label"] == "primary":
                    stt, why = PASS, "validated vs M1, ≥ practical margin" + (", above the within-position shuffled null" if emb else "")
                elif declared in ("validated", "exploratory"):
                    reasons = [x for x, c in ((f"Δ {pr.get('delta_pct'):.2f} % < {margin} % margin" if pr.get("delta_pct") is not None else "", not big),
                                              ("not above the shuffled null" if above is False else ("shuffled null not computed" if above is None and emb else ""), emb and not above),
                                              ("representation without G15 GO (exploratory screen)", r["label"] != "primary"),
                                              ("declared status exploratory", declared == "exploratory")) if c and x]
                    stt, why = EXPLORATORY, "positive but not demonstrated: " + "; ".join(reasons)
                elif declared == "invalid":
                    stt, why = FAIL, "reliably WORSE than M1 (CI upper < 0)"
                else:
                    stt, why = FAIL, "precise null (CI includes 0, half-width ≤ 2 % of RMSE)" if pn else "improvement not demonstrated (CI includes 0)"
                rows.append(_x("G16-C", "incremental information", subj, stt, evidence=f"ΔRMSE {pr.get('delta_rmse')} ({pr.get('delta_pct')} %), status {declared}, "
                               f"folds better {s['folds_better'][0] if s is not None and s.height and 'folds_better' in s.columns else 'n/a'}, shuffled null: {above}",
                               rule="validated (CI > 0, Holm/BH p ≤ 0.05, ≥ 4/5 folds) ∧ Δ ≥ margin ∧ above shuffled null ∧ G15 GO (representations)",
                               observed=pr.get("delta_rmse"), lo=pr.get("delta_lo"), hi=pr.get("delta_hi"),
                               artifact="tables/ladder/paired · tables/ladder/status · tables/controls/summary", explanation=why))
        prim = [r for r in rows if r["gate"] == "G16-C" and r["subject"].startswith("T0 ·")]
        if g("mde") is not None:
            m = g("mde").filter(pl.col("target_id") == "T0") if "target_id" in g("mde").columns else None
            if m is not None and m.height:
                rows.append(_x("G16-C", "detectability on the primary target", "T0", NOT_EVALUATED, evidence=str(m.to_dicts()[:3])[:500],
                               rule="reported with the gate (gates document)", artifact="tables/detectability/mde",
                               explanation="what effect this cohort could detect; not a pass / fail criterion", next_action="report alongside the G16-C reading"))
        sub = [r["status"] for r in prim]
        s_over = PASS if PASS in sub else (EXPLORATORY if EXPLORATORY in sub else (FAIL if sub and all(x in (FAIL, BLOCKED_GATE) for x in sub) and FAIL in sub
                                                                                  else (BLOCKED_GATE if sub and all(x == BLOCKED_GATE for x in sub) else NOT_EVALUATED)))
        rows.append(_x("G16-C", "G16-C overall (primary target T0)", "T0", s_over, evidence=f"{len(prim)} rung row(s) on T0",
                       rule="any rung PASS on the primary target", explanation="task targets are reported but decide nothing here"))
    # ---------------- G16-D cross-domain similarity (per verified representation)
    reg, vc, dmg, cem = g("registry"), g("vs_controls"), g("damage"), g("cem")
    if reg is None:
        rows.append(_x("G16-D", "cross-domain similarity", "all representations", NOT_EVALUATED, explanation="no provenance registry (run E16.B4)"))
    else:
        for r in reg.iter_rows(named=True):
            rep = r["rep"]
            if r["status"] != "VERIFIED_FOLD_SAFE":
                rows.append(_x("G16-D", "cross-domain similarity", rep, BLOCKED_GATE, evidence=r["status"], rule="verified fold-safe provenance",
                               artifact="tables/representation/registry", explanation=r.get("reasons", "")[:400]))
                continue
            if vc is None:
                rows.append(_x("G16-D", "cross-domain similarity", rep, NOT_EVALUATED, explanation="no controlled retrieval tests (run E16.B5)"))
                continue
            best = None
            for arm in getattr(cfg, "retrieval_arms", ("unaligned",)):
                v = vc.filter((pl.col("rep") == rep) & (pl.col("arm") == arm) & (pl.col("scope") == "all"))
                sp = v.filter((pl.col("pool") == "same_position") & pl.col("strongest"))
                tw = v.filter((pl.col("pool") == "twins") & pl.col("strongest"))
                al = v.filter((pl.col("pool") == "all") & (pl.col("control") == "position_only·analytic"))
                a_ok = bool(sp.height and sp["lo"][0] > 0 and (sp["p_holm_strongest"][0] or 1) <= 0.05)
                t_ok = bool(tw.height and tw["lo"][0] > 0)
                p_ok = bool(al.height and al["lo"][0] > 0)
                destructive = None
                if arm != "unaligned" and dmg is not None:
                    dd = dmg.filter((pl.col("rep") == rep) & (pl.col("arm") == arm))
                    destructive = bool(dd["destructive"].any()) if dd.height else None
                rec = {"arm": arm, "a": a_ok, "t": t_ok, "p": p_ok, "destructive": destructive,
                       "obs": sp["delta_mrr"][0] if sp.height else None, "lo": sp["lo"][0] if sp.height else None, "hi": sp["hi"][0] if sp.height else None,
                       "ctrl": sp["control"][0] if sp.height else None}
                if best is None or (a_ok + t_ok + p_ok) > (best["a"] + best["t"] + best["p"]):
                    best = rec
            c_ok = None
            if cem is not None and "rep" in cem.columns:
                cc = cem.filter(pl.col("rep") == rep).group_by("condition").agg(pl.col("same_position_mrr").mean())
                o = cc.filter(pl.col("condition") == "original")["same_position_mrr"]
                mt = cc.filter(pl.col("condition") == "cem_matched")["same_position_mrr"]
                if o.len() and mt.len() and best and best["obs"] is not None:
                    c_ok = bool(np.isfinite(mt[0]) and (o[0] - mt[0]) <= 0.5 * float(best["obs"]))
            core = best is not None and best["a"] and best["t"] and best["p"] and best["destructive"] is not True
            go = str(r.get("g15_decision")) == "GO"
            if core and c_ok and go:
                stt, why = PASS, f"arm {best['arm']}: beats the strongest control in same-position and twin pools, beats position-only, persists after CEM"
            elif core:
                stt, why = EXPLORATORY, "core retrieval criteria pass but " + ("; ".join(x for x, c in (("CEM persistence not evaluated", c_ok is None),
                                                                                                          ("CEM persistence fails", c_ok is False),
                                                                                                          (f"G15 decision {r.get('g15_decision')}", not go)) if c))
            elif best is None:
                stt, why = NOT_EVALUATED, "no controlled test rows for this representation"
            else:
                stt = FAIL
                why = "; ".join(x for x, c in (("same-position ΔMRR vs strongest control not > 0 (Holm)", not best["a"]), ("stopwatch-twin pool not above control", not best["t"]),
                                               ("all-pool not above position-only", not best["p"]), ("alignment destructive", best["destructive"] is True)) if c)
            rows.append(_x("G16-D", "cross-domain similarity", rep, stt, evidence=f"best arm {best['arm'] if best else None}; strongest control {best['ctrl'] if best else None}; "
                           f"same-pos {best['a'] if best else None} · twins {best['t'] if best else None} · position-only {best['p'] if best else None} · "
                           f"CEM {c_ok} · destructive {best['destructive'] if best else None}",
                           rule="gates document G16-D + stopwatch-twin pool + non-destructive alignment", observed=best["obs"] if best else None,
                           lo=best["lo"] if best else None, hi=best["hi"] if best else None,
                           artifact="tables/retrieval/controlled_vs_controls · tables/retrieval/repeat_damage · tables/retrieval/cem", explanation=why,
                           next_action=None if stt != FAIL else "drop cross-domain similarity claims; Combine-only representations may still be screened as predictors"))
    rows.append(_overall(rows, "G16-D"))
    return pl.DataFrame(rows, infer_schema_length=None)


__all__ = ["NEXT", "g16", "g16_extended"]
