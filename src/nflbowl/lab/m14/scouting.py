"""Phase 4 — the Hidden Braking Advantage study; Phase 6 — evidence synthesis and scouting interpretation.

Hidden Braking Advantage (HBA):
1. braking signature: the pre-declared braking-core metrics (``registry.BRAKING_CORE``) in the agility scope whose
   MEASUREMENT status is not ``invalid`` (outcome-free choice); each standardised within position (all athletes,
   no outcome), signed so that + = the hypothesised better direction; composite = mean of the available signed z
   (≥ half of the metrics);
2. stopwatch twins: athletes of the same position with near-identical official tests (within-position z of the
   twin covariates, Euclidean distance ≤ caliper, greedy one-to-one matching on increasing distance). Matching sees
   ONLY official tests and whether a braking signature exists — never an NFL outcome;
3. twin contrast: AFTER matching, the twin with the higher composite vs the lower one on the frozen NFL target and
   on the NFL-window braking metrics (``m14n__*``); mean difference with a pair bootstrap CI, exact sign test, and the
   Spearman of Δcomposite with Δoutcome;
4. hidden variation: within-twin |Δz| of each braking metric vs random within-position pairs — how much braking
   variation official times leave unexplained (descriptive);
5. case studies: pairs chosen by the largest |Δcomposite| among the closest twins (no outcome), outcome revealed after.

Interpretation limits: twins are matched on measured tests only (residual confounding); associations are not causal;
braking metrics are kinematic proxies (no force inference).
"""

from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np

from . import registry as RG

NAN = float("nan")


# ====================================================================== braking signature
def signature_columns(mstatus: Any, scope: str = "agility") -> dict[str, int]:
    """{column: sign} of the braking-core metrics in ``scope`` whose measurement status is not invalid / blocked."""
    import polars as pl

    ok = set(mstatus.filter(~pl.col("measurement_status").is_in(["invalid", "blocked"]))["column"].to_list()) if mstatus is not None and mstatus.height else set()
    out = {}
    for m in RG.BRAKING_CORE:
        c = RG.combine_column(m, scope)
        if c in ok:
            out[c] = RG.BY_NAME[m].sign
    return out


def braking_signature(ath: Any, cols: dict[str, int], min_group: int = 5, min_frac: float = 0.5) -> Any:
    """Within-position signed z of each signature column, within-position percentile, composite and its count."""
    import polars as pl

    d = ath.select("nfl_id", "position_group", *[c for c in cols if c in ath.columns])
    pos = d["position_group"].fill_null("?").cast(pl.String).to_numpy()
    out: dict[str, Any] = {"nfl_id": d["nfl_id"], "position_group": d["position_group"].fill_null("?").cast(pl.String)}
    Zs = []
    for c, sgn in cols.items():
        if c not in d.columns:
            continue
        x = d[c].cast(pl.Float64).to_numpy()
        z = np.full(x.size, NAN)
        pct = np.full(x.size, NAN)
        ok = np.isfinite(x)
        mu0, sd0 = (float(x[ok].mean()), float(x[ok].std())) if ok.sum() > 1 else (NAN, NAN)
        for p in np.unique(pos):
            m = (pos == p) & ok
            mu, sd = (float(x[m].mean()), float(x[m].std())) if m.sum() >= min_group else (mu0, sd0)
            sd = sd if np.isfinite(sd) and sd > 1e-12 else (sd0 if np.isfinite(sd0) and sd0 > 1e-12 else 1.0)
            z[m] = sgn * (x[m] - mu) / sd if sgn != 0 else (x[m] - mu) / sd
            if m.sum() >= 2:
                r = np.argsort(np.argsort(sgn * x[m] if sgn != 0 else x[m]))
                pct[m] = 100 * r / (m.sum() - 1)
        short = c.split("__", 1)[1]
        out[f"z_{short}"] = z
        out[f"pct_{short}"] = pct
        Zs.append(z)
    Z = np.stack(Zs, 1) if Zs else np.zeros((d.height, 0))
    n = np.isfinite(Z).sum(1)
    need = max(1, int(math.ceil(min_frac * Z.shape[1])))
    with np.errstate(invalid="ignore"):
        comp = np.where(n >= need, np.nansum(Z, 1) / np.maximum(n, 1), NAN)
    out["braking_composite"] = comp
    out["n_braking_metrics"] = n
    return pl.DataFrame(out)


# ====================================================================== stopwatch twins
def twin_covariates(ath: Any, candidates: Sequence[str], min_cov: float) -> list[str]:
    return [c for c in candidates if c in ath.columns and float(ath[c].is_not_null().mean()) >= min_cov]


def stopwatch_twins(df: Any, covs: Sequence[str], caliper: float, min_group: int = 6) -> Any:
    """Greedy one-to-one within-position matching on within-position z of the official covariates (distance ≤ caliper).
    ``df`` holds nfl_id, position_group, covs and braking_composite; rows lacking any of them are not matched."""
    import polars as pl

    covs = list(covs)
    d = df.drop_nulls(subset=[*covs, "braking_composite"]).filter(pl.col("braking_composite").is_finite())
    rows = []
    for p in sorted(set(d["position_group"].to_list())):
        g = d.filter(pl.col("position_group") == p)
        if g.height < min_group:
            continue
        X = g.select([pl.col(c).cast(pl.Float64) for c in covs]).to_numpy()
        sd = X.std(0)
        Z = (X - X.mean(0)) / np.where(sd > 1e-12, sd, 1.0)
        ids = g["nfl_id"].to_numpy()
        D = np.sqrt(((Z[:, None, :] - Z[None, :, :]) ** 2).sum(-1))
        iu = np.triu_indices(len(ids), 1)
        order = np.argsort(D[iu], kind="stable")
        used: set[int] = set()
        for k in order:
            a, b = int(iu[0][k]), int(iu[1][k])
            dist = float(D[a, b])
            if dist > caliper:
                break
            if a in used or b in used:
                continue
            used |= {a, b}
            rec = {"position_group": p, "athlete_a": int(ids[a]), "athlete_b": int(ids[b]), "distance": dist}
            for j, c in enumerate(covs):
                rec[f"abs_dz_{c}"] = float(abs(Z[a, j] - Z[b, j]))
                rec[f"diff_{c}"] = float(X[a, j] - X[b, j])
            rows.append(rec)
    return pl.DataFrame(rows, infer_schema_length=None) if rows else pl.DataFrame(schema={"position_group": pl.String, "athlete_a": pl.Int64,
                                                                                            "athlete_b": pl.Int64, "distance": pl.Float64})


def _sign_test(k: int, n: int) -> float:
    """Exact two-sided binomial sign test (p = 0.5)."""
    if n == 0:
        return NAN
    tail = sum(math.comb(n, i) for i in range(0, min(k, n - k) + 1)) / 2 ** n
    return float(min(1.0, 2 * tail))


def twin_contrast(pairs: Any, sig: Any, outcomes: Any, outcome_cols: Sequence[str], n_boot: int, seed: int) -> tuple[Any, Any]:
    """(per-pair long table: hi = higher-composite twin; summary per outcome). Outcomes are joined AFTER matching."""
    import polars as pl

    from ..stats import spearman

    if pairs is None or not pairs.height:
        return pl.DataFrame(), pl.DataFrame()
    comp = dict(zip(sig["nfl_id"].to_list(), sig["braking_composite"].to_list()))
    O = {c: dict(zip(outcomes["nfl_id"].to_list(), outcomes[c].to_list())) for c in outcome_cols if c in outcomes.columns}
    rows = []
    for r in pairs.iter_rows(named=True):
        a, b = r["athlete_a"], r["athlete_b"]
        ca, cb = comp.get(a), comp.get(b)
        if ca is None or cb is None:
            continue
        hi, lo = (a, b) if ca >= cb else (b, a)
        rec = {"position_group": r["position_group"], "athlete_hi": hi, "athlete_lo": lo, "distance": r["distance"], "delta_composite": abs(ca - cb)}
        for c, m in O.items():
            vh, vl = m.get(hi), m.get(lo)
            rec[f"delta_{c}"] = float(vh - vl) if vh is not None and vl is not None and np.isfinite(vh) and np.isfinite(vl) else NAN
        rows.append(rec)
    long = pl.DataFrame(rows, infer_schema_length=None)
    rng = np.random.default_rng(seed)
    summ = []
    for c in O:
        d = long[f"delta_{c}"].to_numpy().astype(float)
        dc = long["delta_composite"].to_numpy().astype(float)
        ok = np.isfinite(d)
        n = int(ok.sum())
        if n < 5:
            summ.append({"outcome": c, "n_pairs": n})
            continue
        dd, cc = d[ok], dc[ok]
        bs = np.array([dd[rng.integers(0, n, n)].mean() for _ in range(n_boot)]) if n >= 10 else np.array([])
        k_pos = int((dd > 0).sum())
        nz = int((dd != 0).sum())
        summ.append({"outcome": c, "n_pairs": n, "mean_diff_hi_minus_lo": float(dd.mean()), "median_diff": float(np.median(dd)),
                     "ci_lo": float(np.quantile(bs, 0.025)) if bs.size else NAN, "ci_hi": float(np.quantile(bs, 0.975)) if bs.size else NAN,
                     "share_hi_better": float(k_pos / nz) if nz else NAN, "sign_test_p": _sign_test(k_pos, nz),
                     "spearman_dcomposite_doutcome": spearman(cc, dd) if n >= 10 else NAN})
    return long, pl.DataFrame(summ, infer_schema_length=None)


def hidden_variation(pairs: Any, sig: Any, n_random: int, seed: int) -> Any:
    """Per signature metric: mean |Δz| within stopwatch twins vs within random same-position pairs (ratio ≈ 1 → the
    official times leave that braking variation essentially unexplained)."""
    import polars as pl

    zc = [c for c in sig.columns if c.startswith("z_")]
    if pairs is None or not pairs.height or not zc:
        return pl.DataFrame()
    Z = {c: dict(zip(sig["nfl_id"].to_list(), sig[c].to_list())) for c in zc}
    pos = dict(zip(sig["nfl_id"].to_list(), sig["position_group"].to_list()))
    by_pos: dict[str, list[int]] = {}
    for i, p in pos.items():
        by_pos.setdefault(p, []).append(i)
    rng = np.random.default_rng(seed)
    rows = []
    for c in [*zc, "braking_composite"]:
        M = Z[c] if c in Z else dict(zip(sig["nfl_id"].to_list(), sig["braking_composite"].to_list()))
        tw = [abs(M[a] - M[b]) for a, b in zip(pairs["athlete_a"].to_list(), pairs["athlete_b"].to_list())
              if M.get(a) is not None and M.get(b) is not None and np.isfinite(M[a]) and np.isfinite(M[b])]
        rnd = []
        for p in pairs["position_group"].to_list():
            pool = [i for i in by_pos.get(p, []) if M.get(i) is not None and np.isfinite(M[i])]
            for _ in range(max(1, n_random // max(1, pairs.height))):
                if len(pool) >= 2:
                    a, b = rng.choice(pool, 2, replace=False)
                    rnd.append(abs(M[int(a)] - M[int(b)]))
        tm, rm = (float(np.mean(tw)) if tw else NAN), (float(np.mean(rnd)) if rnd else NAN)
        rows.append({"metric": c.removeprefix("z_"), "n_twin_pairs": len(tw), "twin_mean_abs_dz": tm, "random_mean_abs_dz": rm,
                     "ratio_twin_to_random": tm / rm if rm and np.isfinite(rm) and rm > 0 else NAN})
    return pl.DataFrame(rows, infer_schema_length=None)


# ====================================================================== case studies
def select_case_studies(contrast: Any, n: int) -> Any:
    """Largest |Δcomposite| among the twin pairs closer than the median distance — outcome columns are NOT used."""
    import polars as pl

    if contrast is None or not contrast.height:
        return pl.DataFrame()
    med = float(contrast["distance"].median())
    return contrast.filter(pl.col("distance") <= med).sort("delta_composite", descending=True).head(n).select(
        "position_group", "athlete_hi", "athlete_lo", "distance", "delta_composite")


def _fmt(v: Any, d: int = 3) -> str:
    if v is None:
        return "—"
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    return "—" if not np.isfinite(f) else f"{f:.{d}f}"


def case_study_markdown(k: int, pair: dict[str, Any], ath: Any, sig: Any, covs: Sequence[str], sig_cols: Sequence[str], outcome_cols: Sequence[str],
                        label: dict[int, str], figure_rel: str | None = None) -> str:
    """Anonymised two-athlete case study (Markdown). Officials and braking signature first; the NFL outcome is revealed
    after the selection, with the interpretation limits."""
    import polars as pl

    hi, lo = pair["athlete_hi"], pair["athlete_lo"]
    A = {r["nfl_id"]: r for r in ath.filter(pl.col("nfl_id").is_in([hi, lo])).iter_rows(named=True)}
    S = {r["nfl_id"]: r for r in sig.filter(pl.col("nfl_id").is_in([hi, lo])).iter_rows(named=True)}
    a, b = label.get(hi, str(hi)), label.get(lo, str(lo))
    L = [f"# Case study {k}: stopwatch twins ({pair['position_group']})", "",
         f"**Athletes:** {a} (higher braking composite) and {b}. **Official-test distance:** {_fmt(pair['distance'], 2)} (within-position z units). "
         "Pair selected by braking difference among the closest twins; NFL outcomes were not used for the selection.", "",
         "## Official Combine tests", "", f"| Test | {a} | {b} |", "|---|---|---|"]
    for c in covs:
        L.append(f"| {c} | {_fmt(A.get(hi, {}).get(c), 2)} | {_fmt(A.get(lo, {}).get(c), 2)} |")
    L += ["", "## Braking signature (agility drills)", "", f"| Metric | {a} | {b} | {a} pct in position | {b} pct in position |", "|---|---|---|---|---|"]
    for c in sig_cols:
        short = c.split("__", 1)[1]
        L.append(f"| {short} | {_fmt(A.get(hi, {}).get(c))} | {_fmt(A.get(lo, {}).get(c))} | {_fmt(S.get(hi, {}).get(f'pct_{short}'), 0)} | "
                 f"{_fmt(S.get(lo, {}).get(f'pct_{short}'), 0)} |")
    L += [f"| **braking composite (z)** | {_fmt(S.get(hi, {}).get('braking_composite'), 2)} | {_fmt(S.get(lo, {}).get('braking_composite'), 2)} | | |", ""]
    n_ev = "m14c_agility__ctx_n_events"
    L.append(f"Agility events used: {a} {_fmt(A.get(hi, {}).get(n_ev))}, {b} {_fmt(A.get(lo, {}).get(n_ev))}.")
    if figure_rel:
        L += ["", f"![event-aligned speed and acceleration]({figure_rel})"]
    L += ["", "## Revealed after selection: NFL movement outcomes", "", f"| Outcome | {a} | {b} |", "|---|---|---|"]
    for c in outcome_cols:
        L.append(f"| {c} | {_fmt(A.get(hi, {}).get(c))} | {_fmt(A.get(lo, {}).get(c))} |")
    L += ["", "## Interpretation limits", "",
          "- Two athletes are an illustration, not evidence; the population-level twin contrast and the held-out transfer tables carry the evidence.",
          "- Braking metrics are kinematic proxies from 10 Hz tracking (no force, power or injury-risk inference).",
          "- Twins are matched on the recorded official tests only; other differences (role, scheme, health, snaps) are not controlled.", ""]
    return "\n".join(L)


# ====================================================================== evidence synthesis
STATUS_TEXT = {
    "validated": ("Supported on held-out athletes of this cohort: {h}.", "Replicate on an independent draft class and in position-specific samples before using as scouting guidance."),
    "exploratory": ("Hypothesis worth testing: {h} (exploratory evidence only).", "Pre-register and test on new athletes; increase trials per athlete to raise reliability."),
    "inconclusive": ("No reliable incremental evidence for: {h}.", "Do not use for decisions; a larger cohort would be needed to detect a small effect."),
    "invalid": ("Not usable: the measurement is unreliable / unstable or the incremental model is reliably worse.", "Revise the event definition or the drill protocol before any further test."),
    "blocked": ("Not measurable with the available data: {r}.", "Requires synchronised multi-player tracking or additional channels."),
}


def evidence_table(avail: Any, mstatus: Any, novelty: Any, single: Any, family: Any) -> Any:
    """One row per metric column (+ blocked metrics): measurement, novelty, single-metric and family predictive status,
    final status, scouting hypothesis and the validation still required. ``single`` / ``family`` = transfer_status rows
    on the claim cohort vs M1."""
    import polars as pl

    MS = {r["column"]: r for r in mstatus.iter_rows(named=True)} if mstatus is not None and mstatus.height else {}
    NV = {r["column"]: r for r in novelty.iter_rows(named=True)} if novelty is not None and novelty.height else {}
    SG = {r["model"].removeprefix("M1+"): r for r in single.iter_rows(named=True)} if single is not None and single.height else {}
    FM = {r["model"].removeprefix("M1+M14_"): r for r in family.iter_rows(named=True)} if family is not None and family.height else {}
    rows = []
    for col, m in MS.items():
        spec = RG.BY_NAME.get(m["metric"])
        nv, sg = NV.get(col, {}), SG.get(col, {})
        fam = FM.get(m["family"], {})
        ms, ss, fs = m["measurement_status"], sg.get("status"), fam.get("status")
        if ms == "invalid":
            final = "invalid"
        elif ss == "validated" and ms == "validated" and nv.get("novel", False):
            final = "validated"
        elif ss in ("validated", "exploratory") or fs in ("validated", "exploratory"):
            final = "exploratory"
        else:
            final = "inconclusive"
        h, v = STATUS_TEXT[final]
        rows.append({"column": col, "metric": m["metric"], "scope": m["scope"], "family": m["family"], "origin_sport": spec.origin_sport if spec else None,
                     "measurement_status": ms, "reliability": m.get("reliability"), "novelty": nv.get("novelty"), "novel": nv.get("novel"),
                     "max_rho_existing": nv.get("max_rho_existing"), "single_metric_status": ss, "single_delta_rmse": sg.get("delta_rmse"),
                     "single_delta_lo": sg.get("delta_lo"), "single_delta_hi": sg.get("delta_hi"), "family_status": fs, "final_status": final,
                     "scouting_hypothesis": h.format(h=spec.hypothesis if spec else col, r=""), "validation_required": v})
    for r in avail.filter(pl.col("status") == "blocked").iter_rows(named=True):
        h, v = STATUS_TEXT["blocked"]
        spec = RG.BY_NAME.get(r["metric"])
        rows.append({"column": None, "metric": r["metric"], "scope": None, "family": r["family"], "origin_sport": spec.origin_sport if spec else None,
                     "measurement_status": "blocked", "final_status": "blocked", "scouting_hypothesis": h.format(h="", r=r["reason"]), "validation_required": v})
    return pl.DataFrame(rows, infer_schema_length=None)


def research_answers(evidence: Any, primary_status: Any, twin_summary: Any) -> list[dict[str, str]]:
    """The seven Phase-6 questions answered from the measured tables only (counts + named metrics)."""
    import polars as pl

    def names(df: Any) -> str:
        v = [f"{r['metric']} ({r['scope']})" if r.get("scope") else r["metric"] for r in df.iter_rows(named=True)]
        return ", ".join(v[:12]) + (" …" if len(v) > 12 else "") if v else "none"

    e = evidence if evidence is not None else pl.DataFrame()
    out = []
    if e.height:
        out.append({"question": "Which features are reliable?", "answer": names(e.filter(pl.col("measurement_status") == "validated"))})
        out.append({"question": "Which features are distinct from conventional tests?",
                    "answer": names(e.filter(pl.col("novel").fill_null(False) & (pl.col("measurement_status") != "invalid")))})
        out.append({"question": "Which metrics predict NFL outcomes (held-out, beyond M1)?",
                    "answer": names(e.filter(pl.col("single_metric_status") == "validated"))})
        out.append({"question": "Which metrics are promising but inconclusive?", "answer": names(e.filter(pl.col("final_status") == "exploratory"))})
        out.append({"question": "Which metrics are blocked or invalid?", "answer": names(e.filter(pl.col("final_status").is_in(["blocked", "invalid"])))})
    if primary_status is not None and primary_status.height:
        txt = "; ".join(f"{r['model']} vs {r['reference']}: {r['status']} (Δ {r['delta_rmse']:+.4f} [{r['delta_lo']:+.4f}, {r['delta_hi']:+.4f}])"
                        for r in primary_status.iter_rows(named=True))
        out.append({"question": "Do the primary M14 models improve held-out prediction beyond M1?", "answer": txt})
    if twin_summary is not None and twin_summary.height and "mean_diff_hi_minus_lo" in twin_summary.columns:
        t = "; ".join(f"{r['outcome']}: {r.get('mean_diff_hi_minus_lo', NAN):+.4f} [{r.get('ci_lo', NAN):+.4f}, {r.get('ci_hi', NAN):+.4f}] (n={r['n_pairs']})"
                      for r in twin_summary.iter_rows(named=True) if r.get("mean_diff_hi_minus_lo") is not None)
        out.append({"question": "Do stopwatch twins with better braking show better NFL movement outcomes?", "answer": t or "not estimable"})
    return out


__all__ = ["STATUS_TEXT", "braking_signature", "case_study_markdown", "evidence_table", "hidden_variation", "research_answers", "select_case_studies",
           "signature_columns", "stopwatch_twins", "twin_contrast", "twin_covariates"]
