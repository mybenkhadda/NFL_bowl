"""Out-of-game reliability of the context definitions (Milestone 07A), coverage trade-off and decision gate. NumPy / Polars.

Halves          sets of WHOLE GAMES of an athlete (random, balanced within athlete; and odd / even game order).
Cross-fitting   the athlete value of half 0 uses context parameters estimated on the windows of half 1, and vice versa:
                the games an athlete is evaluated on never inform the adjustment applied to them. C0 has no parameter.
Estimate        pooled within-position Pearson correlation of the two half values → Spearman–Brown (full length).
Intervals       athlete bootstrap of the odd / even split (which athletes were sampled); ``split_sd`` shows how much
                the estimate moves between random game splits. Neither covers the unobserved context (scheme, opponent).
Exposure        value from the athlete's first N games (or N windows) against the value from his LATER games, both
                cross-fitted: what a short NFL history says about the rest.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import polars as pl

from ..contracts import TableContract
from ..sprint_rescue.stats import seed_for
from .context_outcomes import ALL, DEFINITIONS, ContextConfig, ContextLabels, Progress

S, I, F, B = pl.String(), pl.Int64(), pl.Float64(), pl.Boolean()
NAN = float("nan")

RELIABILITY = TableContract(
    "nfl_context_outcome_reliability", "Out-of-game reliability of every context definition (cross-fitted game halves).",
    {"definition": S, "method": S, "scope": S, "n_athletes": I, "r_half": F, "reliability": F, "ci_lower": F, "ci_upper": F,
     "icc": F, "split_sd": F, "n_splits": I, "fit_isolation": S}, key=("definition", "method", "scope"),
    area="definition × method × scope",
    docs={"method": "random_game_halves_crossfit · odd_even_games_crossfit · first_games:<N> · first_windows:<N>",
          "r_half": "pooled within-position correlation of the two parts",
          "reliability": "Spearman–Brown for half splits; the plain correlation for first-N rows",
          "ci_lower": "95 % athlete bootstrap (odd / even split for the half-split rows)",
          "split_sd": "SD of the estimate over random game splits",
          "fit_isolation": "where the context parameters of each part were estimated"})
TRADEOFF = TableContract(
    "nfl_context_coverage_tradeoff", "Reliability, coverage and agreement with the raw outcome, by context complexity.",
    {"definition": S, "label": S, "complexity": I, "status": S, "n_athletes_eligible": I, "share_athletes_retained": F,
     "n_positions": I, "reliability": F, "ci_lower": F, "ci_upper": F, "min_position_reliability": F,
     "spearman_with_c0": F, "feasible": B, "reason": S}, key=("definition",), area="definition",
    docs={"share_athletes_retained": "eligible athletes / eligible athletes of C0",
          "spearman_with_c0": "within-position rank agreement with the raw aggregate (athletes eligible under both)",
          "feasible": "reference lines met (heuristics): reliability, retention, position coverage"})
GATE = TableContract("context_gate", "Decision gate of a Milestone 07 notebook.", {"criterion": S, "passed": B, "detail": S},
                     key=("criterion",), area="criterion")


# ============================================================================== helpers
def game_halves(lab: ContextLabels, rng: np.random.Generator | None) -> np.ndarray:
    """(W,) half 0 / 1 of every window; a half is a set of whole games. ``rng`` None → parity of the game order."""
    w = lab.w
    if rng is None:
        return (w.g_order % 2).astype(np.int64)
    aa = w.ag_athlete
    o = np.lexsort((rng.random(aa.size), aa))
    start = np.r_[0, np.flatnonzero(np.diff(aa[o])) + 1]
    rank = np.arange(aa.size) - np.repeat(start, np.diff(np.r_[start, aa.size]))
    half = np.empty(aa.size, dtype=np.int64)
    half[o] = (rank + rng.integers(0, 2, size=aa.size)[aa[o]]) % 2          # random start parity per athlete
    return half[w.ag]


def _centre(v: np.ndarray, pos: np.ndarray) -> np.ndarray:
    out = v.astype(float).copy()
    for p in np.unique(pos):
        k = pos == p
        out[k] -= out[k].mean()
    return out


def pair_stats(x: np.ndarray, y: np.ndarray, pos: np.ndarray) -> dict[str, float]:
    """Pooled within-position correlation of two parts, its Spearman–Brown projection, and ICC(1) of the pair."""
    if x.size < 3:
        return {"r": NAN, "sb": NAN, "icc": NAN}
    a, b = _centre(x, pos), _centre(y, pos)
    den = float(np.sqrt((a ** 2).sum() * (b ** 2).sum()))
    r = float((a * b).sum() / den) if den > 1e-12 else NAN
    m = (a + b) / 2
    msb = float(2 * (m ** 2).sum() / max(x.size - np.unique(pos).size, 1))
    msw = float(((a - m) ** 2 + (b - m) ** 2).sum() / x.size)
    return {"r": r, "sb": 2 * r / (1 + r) if np.isfinite(r) and r > -1 else NAN,
            "icc": (msb - msw) / (msb + msw) if msb + msw > 0 else NAN}


def split_values(lab: ContextLabels, definition: str, part0: np.ndarray, part1: np.ndarray, min_n: tuple[int, int]
                 ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(value of part 0, value of part 1, athletes with enough windows in both). Cross-fitted: part 0 is adjusted with
    parameters from part 1 and vice versa."""
    v0 = lab.values(definition, fit=part1, score=part0)
    v1 = lab.values(definition, fit=part0, score=part1)
    ok = (v0["n_windows"] >= min_n[0]) & (v1["n_windows"] >= min_n[1]) & np.isfinite(v0["value"]) & np.isfinite(v1["value"])
    return v0["value"], v1["value"], ok


def _boot(x: np.ndarray, y: np.ndarray, pos: np.ndarray, key: str, n_boot: int, rng: np.random.Generator) -> tuple[float | None, float | None]:
    if x.size < 10 or n_boot <= 0:
        return None, None
    reps = np.array([pair_stats(x[i], y[i], pos[i])[key] for i in rng.integers(0, x.size, size=(n_boot, x.size))])
    ok = reps[np.isfinite(reps)]
    return (float(np.quantile(ok, 0.025)), float(np.quantile(ok, 0.975))) if ok.size >= 20 else (None, None)


# ============================================================================== reliability across games
def reliability_table(lab: ContextLabels, definitions: Sequence[str], seed: int, progress: bool = True) -> pl.DataFrame:
    """Random game half-splits (cross-fitted) and the odd / even split with athlete-bootstrap intervals, pooled within
    position and by position. Only athletes meeting the exposure requirement of the definition take part."""
    w, cc = lab.w, lab.cc
    defs = [d for d in definitions if lab.available(d)[0]]
    rows: list[dict[str, Any]] = []
    bar = Progress(len(defs) * (cc.n_game_splits + 1), "07A reliability", every=max(1, cc.n_game_splits // 5)) if progress else None
    scopes = (ALL, *w.pos_levels.tolist())
    for d in defs:
        elig = lab.eligible(d)
        iso = "no parameter" if d == "C0" else "parameters from the other game half (cross-fitted)"
        rng = np.random.default_rng(seed_for(seed, "ctx", "halves", d))
        sb: dict[str, list[float]] = {s: [] for s in scopes}
        rr: dict[str, list[float]] = {s: [] for s in scopes}
        for _ in range(cc.n_game_splits):
            h = game_halves(lab, rng)
            x, y, ok = split_values(lab, d, h == 0, h == 1, (cc.min_half_windows, cc.min_half_windows))
            for s in scopes:
                k = ok & elig & (np.ones(w.n_athletes, dtype=bool) if s == ALL else w.position == s)
                st = pair_stats(x[k], y[k], w.pos_a[k])
                sb[s].append(st["sb"])
                rr[s].append(st["r"])
            if bar:
                bar.step(note=d)
        h = game_halves(lab, None)
        x, y, ok = split_values(lab, d, h == 0, h == 1, (cc.min_half_windows, cc.min_half_windows))
        brng = np.random.default_rng(seed_for(seed, "ctx", "boot", d))
        for s in scopes:
            k = ok & elig & (np.ones(w.n_athletes, dtype=bool) if s == ALL else w.position == s)
            n = int(k.sum())
            if n < (cc.min_athletes if s == ALL else cc.min_athletes_position):
                continue
            st = pair_stats(x[k], y[k], w.pos_a[k])
            lo, hi = _boot(x[k], y[k], w.pos_a[k], "sb", cc.n_boot, brng)
            a_sb, a_r = np.array(sb[s], dtype=float), np.array(rr[s], dtype=float)
            fin = a_sb[np.isfinite(a_sb)]
            rows.append({"definition": d, "method": "random_game_halves_crossfit", "scope": s, "n_athletes": n,
                         "r_half": float(np.nanmean(a_r)) if np.isfinite(a_r).any() else None,
                         "reliability": float(fin.mean()) if fin.size else None, "ci_lower": lo, "ci_upper": hi, "icc": st["icc"],
                         "split_sd": float(fin.std(ddof=1)) if fin.size > 1 else None, "n_splits": int(fin.size), "fit_isolation": iso})
            rows.append({"definition": d, "method": "odd_even_games_crossfit", "scope": s, "n_athletes": n, "r_half": st["r"],
                         "reliability": st["sb"], "ci_lower": lo, "ci_upper": hi, "icc": st["icc"], "split_sd": None, "n_splits": 1,
                         "fit_isolation": iso})
        if bar:
            bar.step(note=f"{d} odd/even + bootstrap")
    out = pl.DataFrame(rows, schema=RELIABILITY.columns) if rows else RELIABILITY.empty()
    return out.with_columns([pl.col(c).fill_nan(None) for c in ("r_half", "reliability", "icc", "split_sd")])


def exposure_table(lab: ContextLabels, definitions: Sequence[str], seed: int) -> pl.DataFrame:
    """First N games / first N windows against the athlete's LATER games (≥ ``holdout_min_windows`` windows there).
    Each part is adjusted with parameters from the other part. The cohort changes with N: read ``n_athletes``."""
    w, cc = lab.w, lab.cc
    rows: list[dict[str, Any]] = []
    for d in definitions:
        if not lab.available(d)[0]:
            continue
        iso = "no parameter" if d == "C0" else "first part ← parameters from later games, and vice versa"
        rng = np.random.default_rng(seed_for(seed, "ctx", "exposure", d))
        plans = [(f"first_games:{g}", w.g_order <= g, w.g_order > g, 1) for g in cc.first_games_grid]
        for n_first in cc.first_windows_grid:
            last = np.zeros(w.n_athletes, dtype=np.int64)
            np.maximum.at(last, w.a[w.w_order <= n_first], w.g_order[w.w_order <= n_first])
            enough = np.bincount(w.a, minlength=w.n_athletes) >= n_first
            plans.append((f"first_windows:{n_first}", (w.w_order <= n_first) & enough[w.a], (w.g_order > last[w.a]) & enough[w.a], n_first))
        for method, first, later, need in plans:
            if not first.any() or not later.any():
                continue
            x, y, ok = split_values(lab, d, first, later, (min(need, cc.min_half_windows) if method.startswith("first_games") else
                                                           max(1, int(0.8 * need)), cc.holdout_min_windows))
            n = int(ok.sum())
            if n < cc.min_athletes:
                continue
            st = pair_stats(x[ok], y[ok], w.pos_a[ok])
            lo, hi = _boot(x[ok], y[ok], w.pos_a[ok], "r", min(cc.n_boot, 500), rng)
            rows.append({"definition": d, "method": method, "scope": ALL, "n_athletes": n, "r_half": st["r"], "reliability": st["r"],
                         "ci_lower": lo, "ci_upper": hi, "icc": None, "split_sd": None, "n_splits": 1, "fit_isolation": iso})
    out = pl.DataFrame(rows, schema=RELIABILITY.columns) if rows else RELIABILITY.empty()
    return out.with_columns([pl.col(c).fill_nan(None) for c in ("r_half", "reliability")])


# ============================================================================== trade-off and gate
def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra, rb = np.argsort(np.argsort(a)).astype(float), np.argsort(np.argsort(b)).astype(float)
    ra, rb = ra - ra.mean(), rb - rb.mean()
    den = float(np.sqrt((ra ** 2).sum() * (rb ** 2).sum()))
    return float((ra * rb).sum() / den) if den > 1e-12 else NAN


def tradeoff_table(lab: ContextLabels, overlap: pl.DataFrame, reliability: pl.DataFrame, definitions: Sequence[str]) -> pl.DataFrame:
    """One row per definition: complexity, coverage kept, cross-fitted reliability and rank agreement with C0.
    ``feasible`` applies the reference lines of the configuration — heuristics that inform the analyst's reading."""
    w, cc = lab.w, lab.cc
    ov = {r["definition"]: r for r in overlap.iter_rows(named=True)}
    rel = reliability.filter(pl.col("method") == "random_game_halves_crossfit")
    base = lab.values("C0")
    e0 = base["n_windows"] >= cc.min_windows
    rows = []
    for d in definitions:
        o = ov.get(d)
        if o is None or o["status"] == "unavailable":
            continue
        r_all = rel.filter((pl.col("definition") == d) & (pl.col("scope") == ALL))
        r_pos = rel.filter((pl.col("definition") == d) & (pl.col("scope") != ALL))
        v = lab.values(d)
        both = e0 & (v["n_windows"] >= cc.min_windows) & np.isfinite(v["value"]) & np.isfinite(base["value"])
        rho = _spearman(_centre(base["value"][both], w.pos_a[both]), _centre(v["value"][both], w.pos_a[both])) if both.sum() >= 5 else NAN
        r = r_all["reliability"][0] if r_all.height else None
        kept = o["n_athletes_eligible"] / max(int(e0.sum()), 1)
        fails = ([f"reliability {r if r is None else round(r, 3)} < {cc.ref_reliability}"] if r is None or r < cc.ref_reliability else []) \
            + ([f"keeps {kept:.0%} of the C0 cohort < {cc.ref_retention:.0%}"] if kept < cc.ref_retention else []) \
            + ([f"only {o['n_positions']} position group(s) with ≥ {cc.min_athletes_position} athletes"] if o["n_positions"] < 3 else [])
        rows.append({"definition": d, "label": DEFINITIONS[d]["label"], "complexity": DEFINITIONS[d]["complexity"],
                     "status": DEFINITIONS[d]["status"], "n_athletes_eligible": o["n_athletes_eligible"], "share_athletes_retained": kept,
                     "n_positions": o["n_positions"], "reliability": r, "ci_lower": r_all["ci_lower"][0] if r_all.height else None,
                     "ci_upper": r_all["ci_upper"][0] if r_all.height else None,
                     "min_position_reliability": r_pos["reliability"].min() if r_pos.height else None,
                     "spearman_with_c0": rho if np.isfinite(rho) else None, "feasible": not fails, "reason": "; ".join(fails)})
    return pl.DataFrame(rows, schema=TRADEOFF.columns) if rows else TRADEOFF.empty()


def gate_07a(overlap: pl.DataFrame, variance: pl.DataFrame, scale: pl.DataFrame, reliability: pl.DataFrame, tradeoff: pl.DataFrame,
             cc: ContextConfig) -> pl.DataFrame:
    """Gate 07A — context outcome feasibility. PASS = the question was answered validly; it does not require that an
    adjusted definition beats the raw one. Outcome PROMOTION needs ≥ 1 exploratory definition marked feasible."""
    rows = []

    def add(name: str, ok: bool, detail: str) -> None:
        rows.append({"criterion": name, "passed": bool(ok), "detail": detail})

    avail = overlap.filter(pl.col("status") != "unavailable")
    add("definitions_documented", avail.height >= 2 and overlap.height >= 1,
        f"{avail.height} definition(s) computed; unavailable: {overlap.filter(pl.col('status') == 'unavailable')['definition'].to_list()}")
    ath = variance.filter((pl.col("scope") == ALL) & (pl.col("component") == "athlete"))
    share = ath["share_within_position"][0] if ath.height else None
    snr = scale.filter((pl.col("scope") == ALL) & (pl.col("definition") == "C0"))
    add("athlete_variation_retained", share is not None and share > 0 and snr.height > 0 and (snr["snr"][0] or 0) > 0,
        f"athlete share of the within-position window variance {share}; between-athlete SD / standard error "
        f"{snr['snr'][0] if snr.height else None} (reference {cc.ref_snr})")
    cross = reliability.filter((pl.col("method") == "random_game_halves_crossfit") & (pl.col("scope") == ALL))
    add("reliability_evaluated_across_games", cross.height >= 1 and cross["reliability"].null_count() == 0,
        "; ".join(f"{r['definition']} {r['reliability']:.3f} (n = {r['n_athletes']})" for r in cross.iter_rows(named=True) if r["reliability"] is not None))
    add("coverage_documented", avail.height > 0 and avail["n_athletes_eligible"].null_count() == 0,
        "; ".join(f"{r['definition']} {r['n_athletes_eligible']} athletes" for r in avail.iter_rows(named=True)))
    low = avail.filter(pl.col("n_positions") < 3)["definition"].to_list()
    add("opportunity_overlap_adequate", avail.height > 0 and len(low) < avail.height,
        "every definition covers ≥ 3 position groups" if not low else f"fewer than 3 position groups for {low}")
    promoted = tradeoff.filter(pl.col("feasible") & (pl.col("status") == "exploratory"))["definition"].to_list()
    add("promotion_candidates (informative)", True,
        f"exploratory definitions meeting the reference lines: {promoted or 'none — keep the frozen target, promote nothing'}")
    return pl.DataFrame(rows, schema=GATE.columns)


def assert_gate(gate: pl.DataFrame, name: str) -> None:
    failed = gate.filter(~pl.col("passed"))
    if failed.height:
        raise RuntimeError(f"Gate {name} NOT passed:\n" + "\n".join(f"  ✗ {r['criterion']}: {r['detail']}" for r in failed.iter_rows(named=True)))


def conclusions_07a(variance: pl.DataFrame, tradeoff: pl.DataFrame, scale: pl.DataFrame, outcome_id: str, unit: str) -> str:
    """Markdown reading of the executed tables. Every number comes from the arguments."""
    f = lambda x, d=3: "n/a" if x is None else f"{x:.{d}f}"
    v = {r["component"]: r for r in variance.filter(pl.col("scope") == ALL).iter_rows(named=True)}
    lines = [f"### What the tables show (`{outcome_id}`, {unit})", ""]
    if "position" in v:
        lines.append(f"- **Position** carries {f(v['position']['share_total'])} of all window variance — a fixed label, not an athlete difference.")
    for k, name in (("athlete", "stable athlete differences"), ("game_within_athlete", "game-to-game variation of an athlete"),
                    ("window_within_game", "window-to-window variation inside a game")):
        if k in v:
            r = v[k]
            lines.append(f"- Within position, **{name}**: {f(r['share_within_position'])} [{f(r['ci_lower'])}, {f(r['ci_upper'])}]"
                         + (" — truncated at 0" if r["truncated"] else ""))
    for k, r in v.items():
        if k.startswith("context:"):
            lines.append(f"- Marginal share of `{k[8:]}` within position: {f(r['share_within_position'])} (window level, not additive).")
    s = scale.filter((pl.col("scope") == ALL) & (pl.col("definition") == "C0"))
    if s.height:
        lines.append(f"- Scale: between-athlete SD {f(s['between_sd'][0])} {unit} within position; standard error of one athlete's value "
                     f"{f(s['standard_error'][0])} {unit} at the median exposure (ratio {f(s['snr'][0], 2)}).")
    lines += ["", "| Definition | Athletes | Kept | Reliability [95 %] | ρ with C0 | Feasible |", "|---|---|---|---|---|---|"]
    for r in tradeoff.sort("complexity").iter_rows(named=True):
        lines.append(f"| {r['definition']} — {r['label']} | {r['n_athletes_eligible']} | {f(r['share_athletes_retained'], 2)} | "
                     f"{f(r['reliability'])} [{f(r['ci_lower'])}, {f(r['ci_upper'])}] | {f(r['spearman_with_c0'])} | "
                     f"{'yes' if r['feasible'] else 'no — ' + r['reason']} |")
    lines += ["", "Limits: an adjustment removes what OBSERVED labels explain (alignment, play class, starting speed, season, down). "
              "Assignment, opponent and scheme are unobserved; role is partly chosen because of ability. A context-adjusted value is "
              "still observed movement — not athletic capability, and not football performance. Every C1–C4 definition is EXPLORATORY."]
    return "\n".join(lines)
