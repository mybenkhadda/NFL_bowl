"""Figures of notebooks 05A (NFL movement outcome EDA) and 05B (game-grouped reliability and target freeze). Every
function returns a Matplotlib figure; the notebook passes it to ``NflOutcomeStore.show`` (inline display, PNG 300 DPI +
SVG, PDF for publication figures).

Conventions of the programme: project 16:9 style, units on every axis, the unit of analysis and the denominator in the
caption. A frame or a window is never called a sample size: counts of athletes are printed beside counts of windows.
Colour identifies the POSITION GROUP in 05A and the outcome FAMILY (O1–O6) in 05B. No Combine value is drawn here.
"""

from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np
import polars as pl

from ...data.reporting import CATEGORICAL, INK, SEQUENTIAL_BLUE, STATUS
from ...features.play_windows import PP_KEYS
from ..figkit import empty_panel, figure, finish
from .nfl_outcomes import ALL, STAGES, OutcomeConfig, blab, wlab
from .outcome_reliability import WITHIN

FAMILY_COLOR = {f"O{i + 1}": CATEGORICAL[i] for i in range(6)}
STATUS_COLOR = {"eligible": STATUS["good"], "provisional": CATEGORICAL[0], "not_eligible": STATUS["critical"],
                "component": INK["axis"], "reference_only": INK["muted"], "exploratory": STATUS["warning"]}
REASON_COLOR = {"": STATUS["good"], "short_post_snap_window": CATEGORICAL[3], "quality_tier_not_usable": CATEGORICAL[1],
                "no_quality_record": CATEGORICAL[4], "nullified_by_penalty": CATEGORICAL[6], "position_out_of_scope": INK["axis"]}
_NAMES = {"speed_max": "single-frame peak speed", "speed_p90": "play p90 speed", "speed_mean": "mean post-snap speed",
          "decel_sust": "sustained braking", "gain_d": "speed gain over the distance window", "reaccel_gain": "re-acceleration gain",
          "brake_v0": "speed before braking", "gain_d_time": "time over the distance window"}


# ============================================================================== helpers
def _num(x: Any) -> float:
    return float("nan") if x is None else float(x)


def _short(oid: str) -> str:
    return oid.split(".", 1)[-1]


def _positions(df: pl.DataFrame, col: str = "position_group") -> list[str]:
    return sorted(p for p in df[col].drop_nulls().unique().to_list() if p not in (ALL, WITHIN))


def _pos_color(positions: Sequence[str]) -> dict[str, str]:
    return {p: CATEGORICAL[i % len(CATEGORICAL)] for i, p in enumerate(positions)}


def _cmap(colors: Sequence[str]):
    from matplotlib.colors import LinearSegmentedColormap

    cm = LinearSegmentedColormap.from_list("nfl_outcomes", list(colors))
    cm.set_bad(INK["grid"])
    return cm


def _heat(ax, M: np.ndarray, rows: Sequence[str], cols: Sequence[str], fmt: str = "{:.2f}", colors: Sequence[str] | None = None,
          vmin: float = 0.0, vmax: float = 1.0, fontsize: float = 8, labels: np.ndarray | None = None) -> None:
    """Annotated matrix; missing cells are grey and print nothing."""
    M = np.asarray(M, dtype=float).reshape(len(rows), len(cols))
    ax.imshow(np.ma.masked_invalid(M), cmap=_cmap(colors or ["#ffffff", SEQUENTIAL_BLUE[4]]), vmin=vmin, vmax=vmax, aspect="auto")
    for i in range(len(rows)):
        for j in range(len(cols)):
            if labels is not None:
                ax.text(j, i, str(labels[i][j]), ha="center", va="center", fontsize=fontsize, color=INK["primary"])
            elif np.isfinite(M[i, j]):
                ax.text(j, i, fmt.format(M[i, j]), ha="center", va="center", fontsize=fontsize, color=INK["primary"])
    ax.set_xticks(np.arange(len(cols)))
    ax.set_xticklabels(list(cols), fontsize=8)
    ax.set_yticks(np.arange(len(rows)))
    ax.set_yticklabels(list(rows), fontsize=8)
    ax.grid(False)


def measure_label(col: str) -> str:
    """Axis label with the unit of a play measure."""
    unit = "s" if col.endswith("_time") else "yd/s²" if col.startswith(("accel", "band_acc", "decel")) else "yd/s"
    if col in _NAMES:
        name = _NAMES[col]
    elif col.startswith("speed_sust"):
        name = f"sustained speed, {int(col[-2:]) / 10:g} s"
    elif col.startswith("accel_sust"):
        name = f"sustained acceleration, {int(col[-2:]) / 10:g} s"
    elif col.startswith("gain") and col[4:].isdigit():
        name = f"speed gain over {int(col[4:]) / 10:g} s"
    elif col.startswith("band_acc_"):
        name = "acceleration from " + col[len("band_acc_v"):].replace("_", "–").replace("p", ".") + " yd/s"
    elif col.startswith("band_gain_"):
        name = "largest speed gain from " + col[len("band_gain_v"):].replace("_", "–").replace("p", ".") + " yd/s"
    else:
        name = col
    return f"{name} ({unit})"


def _box_stats(df: pl.DataFrame, col: str, by: str = "position_group") -> list[dict[str, Any]]:
    """Quantiles of ``col`` per group for a box drawn without loading the windows into Matplotlib (5–95 % whiskers)."""
    return (df.filter(pl.col(col).is_not_null()).group_by(by)
            .agg(pl.len().alias("n"), pl.col("nfl_id").n_unique().alias("a"),
                 *[pl.col(col).quantile(q, "linear").alias(k)
                   for k, q in (("whislo", 0.05), ("q1", 0.25), ("med", 0.5), ("q3", 0.75), ("whishi", 0.95))])
            .sort(by).to_dicts())


def _boxes(ax, stats: Sequence[dict[str, Any]], colors: dict[str, str], by: str = "position_group") -> None:
    if not stats:
        empty_panel(ax, "no eligible window with a value")
        return
    clean = [{k: float(s[k]) for k in ("whislo", "q1", "med", "q3", "whishi")} | {"fliers": []} for s in stats]
    bp = ax.bxp(clean, positions=np.arange(len(stats)), showfliers=False, patch_artist=True, widths=0.6,
                medianprops={"color": INK["primary"], "linewidth": 1.5})
    for patch, s in zip(bp["boxes"], stats):
        patch.set_facecolor(colors.get(s[by], INK["axis"]))
        patch.set_alpha(0.85)
    ax.set_xticks(np.arange(len(stats)))
    ax.set_xticklabels([f"{s[by]}\n{s['n']:,} w\n{s['a']} ath" for s in stats], fontsize=8)


def _eligible(plays: pl.DataFrame) -> pl.DataFrame:
    return plays.filter(pl.col("eligible"))


def measure_summary(plays: pl.DataFrame, columns: Sequence[str], limits: dict[str, float] | None = None) -> pl.DataFrame:
    """Windows, athletes and quantiles of each play measure by position group (eligible windows). ``limits``: measure →
    plausibility limit; ``share_above_limit`` is the share of defined windows beyond it (null without a limit)."""
    el, parts = _eligible(plays), []
    for k in columns:
        ok = pl.col(k).is_not_null()
        lim = (limits or {}).get(k)
        parts.append(el.group_by("position_group").agg(
            ok.sum().cast(pl.Int64).alias("windows"), pl.col("nfl_id").filter(ok).n_unique().cast(pl.Int64).alias("athletes"),
            pl.col(k).quantile(0.1, "linear").alias("p10"), pl.col(k).median().alias("median"), pl.col(k).quantile(0.9, "linear").alias("p90"),
            pl.col(k).quantile(0.99, "linear").alias("p99"), pl.col(k).max().alias("max"),
            ((pl.col(k) > (float("inf") if lim is None else lim)).sum() / ok.sum()).alias("share_above_limit"),
        ).with_columns(pl.lit(k).alias("measure"), pl.lit(measure_label(k).rsplit("(", 1)[-1].rstrip(")")).alias("unit"),
                       pl.when(pl.lit(lim is None)).then(None).otherwise(pl.col("share_above_limit")).alias("share_above_limit")))
    schema = {"measure": pl.String, "unit": pl.String, "position_group": pl.String, "windows": pl.Int64, "athletes": pl.Int64,
              "p10": pl.Float64, "median": pl.Float64, "p90": pl.Float64, "p99": pl.Float64, "max": pl.Float64, "share_above_limit": pl.Float64}
    if not parts:
        return pl.DataFrame(schema=schema)
    return pl.concat(parts, how="vertical").select([pl.col(n).cast(t) for n, t in schema.items()]).sort("measure", "position_group")


def _pick(rel: pl.DataFrame, oid: str, method: str, c: OutcomeConfig, adj: str = "raw", group: str = WITHIN) -> dict[str, Any] | None:
    r = rel.filter((pl.col("outcome_id") == oid) & (pl.col("method") == method) & (pl.col("adjustment") == adj)
                   & (pl.col("group") == group) & (pl.col("min_windows") == c.primary_min_windows))
    return r.row(0, named=True) if r.height else None


# ============================================================================== 05A · 1 cohort
def plot_cohort(plays: pl.DataFrame, c: OutcomeConfig):
    el = _eligible(plays)
    fig, ax = figure(1, 2)
    positions = _positions(el)
    col = _pos_color(positions)
    for axis, key, lab in ((ax[0][0], "season", "NFL season"), (ax[0][1], "cohort", "draft cohort")):
        t = el.filter(pl.col(key).is_not_null()).group_by("position_group", key).agg(pl.col("nfl_id").n_unique().alias("n"))
        if not t.height:
            empty_panel(axis, "no eligible window")
            continue
        levels = sorted(t[key].unique().to_list())
        got = {(r["position_group"], r[key]): r["n"] for r in t.iter_rows(named=True)}
        w = 0.8 / max(len(positions), 1)
        for i, p in enumerate(positions):
            v = [got.get((p, lv), 0) for lv in levels]
            x = np.arange(len(levels)) + (i - (len(positions) - 1) / 2) * w
            axis.bar(x, v, width=w, color=col[p], label=p)
            for xi, vi in zip(x, v):
                axis.text(xi, vi, str(vi), ha="center", va="bottom", fontsize=7, color=INK["secondary"])
        axis.set_xticks(np.arange(len(levels)))
        axis.set_xticklabels([str(lv) for lv in levels])
        axis.set_xlabel(lab)
        axis.set_ylabel("athletes with ≥ 1 eligible window")
        axis.set_title(f"By position group and {lab}", fontsize=11)
    ax[0][0].legend(frameon=False, fontsize=8, title="position group")
    n = el["nfl_id"].n_unique() if el.height else 0
    finish(fig, f"Unit: athlete. Denominator: {n} athletes with at least one eligible post-snap window (season types "
                f"{', '.join(c.season_types)}; quality tier in {list(c.usable_tiers)}; ≥ {c.min_post_frames} valid post-snap frames). An athlete "
                "tracked in several seasons is counted once per season on the left and once in the cohort panel. NFL tracking covers the "
                "Combine cohorts only, so a later cohort has fewer seasons of exposure by construction.",
           "NFL tracking cohort by position, season and draft cohort")
    return fig


# ============================================================================== 05A · 2 eligible windows
def plot_window_counts(funnel: pl.DataFrame, plays: pl.DataFrame, c: OutcomeConfig):
    fig, ax = figure(1, 2)
    a, b = ax[0]
    f = {r["stage"]: r for r in funnel.filter(pl.col("position_group") == ALL).iter_rows(named=True)}
    stages = [s for s in STAGES if s in f]
    if stages:
        v = [f[s]["n_player_plays"] for s in stages]
        a.barh(np.arange(len(stages)), v, color=SEQUENTIAL_BLUE[3])
        for i, s in enumerate(stages):
            a.text(v[i], i, f"  {v[i]:,} player-plays · {f[s]['n_athletes']} athletes · {f[s]['n_athlete_games']:,} athlete-games",
                   va="center", fontsize=8, color=INK["secondary"])
        a.set_yticks(np.arange(len(stages)))
        a.set_yticklabels([s.replace("_", " ") for s in stages], fontsize=9)
        a.invert_yaxis()
        a.set_xlim(0, max(v) * 1.9)
        a.set_xlabel("tracked player-plays remaining")
    else:
        empty_panel(a, "no funnel row")
    a.set_title("Eligibility funnel (rules applied in order)", fontsize=11)
    el = _eligible(plays)
    per = el.group_by("nfl_id").agg(pl.len().alias("n"), pl.col("position_group").first())
    positions = _positions(per)
    col = _pos_color(positions)
    for p in positions:
        n = np.sort(per.filter(pl.col("position_group") == p)["n"].to_numpy())
        b.step(n, 1.0 - np.arange(n.size) / n.size, where="post", color=col[p], label=f"{p} ({n.size} athletes)")
    for t in c.min_windows_grid:
        b.axvline(t, color=INK["axis"], linewidth=0.8, linestyle="--")
        b.text(t, 1.01, str(t), ha="center", va="bottom", fontsize=7, color=INK["muted"])
    if positions:
        b.set_xscale("log")
        b.legend(frameon=False, fontsize=8)
    else:
        empty_panel(b, "no eligible window")
    b.set_xlabel("eligible windows of the athlete (log axis)")
    b.set_ylabel("share of athletes with at least that many")
    b.set_title("Exposure per athlete", fontsize=11)
    finish(fig, "Left, unit: tracked player-play with a snap tag; each bar keeps the rows that pass every rule above it. Right, unit: "
                "athlete; one eligible window = one player-play post-snap window. Dashed lines: the exposure thresholds of the "
                f"reliability study ({list(c.min_windows_grid)} windows). Exposure differs between athletes by an order of magnitude: an "
                "athlete statistic that follows the window count measures usage, not movement.",
           "Eligible movement windows")
    return fig


# ============================================================================== 05A · 3–4 distributions
def plot_measure_distributions(plays: pl.DataFrame, columns: Sequence[str], c: OutcomeConfig, title: str, note: str = ""):
    el = _eligible(plays)
    fig, ax = figure(1, len(columns))
    col = _pos_color(_positions(el))
    for axis, k in zip(ax[0], columns):
        _boxes(axis, _box_stats(el, k), col)
        axis.set_ylabel(measure_label(k))
        axis.set_title(measure_label(k).rsplit(" (", 1)[0], fontsize=10)
    finish(fig, "Unit: eligible player-play window (w); athletes (ath) printed under each group — windows of one athlete are not "
                "independent. Box: quartiles and median; whiskers: 5th and 95th percentiles of the windows. Values are play measures from "
                f"derived kinematics (Savitzky–Golay window {c.primary_smoothing[0]} frames, order {c.primary_smoothing[1]}), post-snap "
                "frames only. " + note, title)
    return fig


# ============================================================================== 05A · 5 phase space
def plot_phase_space(hist: pl.DataFrame, c: OutcomeConfig, min_frames: int = 200):
    positions = _positions(hist)
    fig, ax = figure(1, max(len(positions), 1))
    nb_s, nb_a = int(round(c.hist_speed_max / c.hist_speed_step)), int(round(2 * c.hist_accel_lim / c.hist_accel_step))
    centres = (np.arange(nb_s) + 0.5) * c.hist_speed_step
    if not positions:
        empty_panel(ax[0][0], "no frame")
    for axis, p in zip(ax[0], positions):
        sub = hist.filter(pl.col("position_group") == p)
        M = np.zeros((nb_a, nb_s))
        M[np.clip(sub["accel_bin"].to_numpy(), 0, nb_a - 1), np.clip(sub["speed_bin"].to_numpy(), 0, nb_s - 1)] = sub["n_frames"].to_numpy()
        tot = M.sum(0)
        ok = tot >= min_frames
        with np.errstate(divide="ignore", invalid="ignore"):
            P = np.where(ok[None, :], M / np.where(tot > 0, tot, 1.0), np.nan)
            cdf = np.cumsum(M, 0) / np.where(tot > 0, tot, 1.0)
        axis.imshow(np.ma.masked_invalid(P), origin="lower", aspect="auto", cmap=_cmap(["#ffffff", SEQUENTIAL_BLUE[5]]),
                    extent=(0.0, c.hist_speed_max, -c.hist_accel_lim, c.hist_accel_lim), vmin=0.0)
        for q, style in ((0.5, "-"), (0.9, "--")):
            edge = -c.hist_accel_lim + ((cdf >= q).argmax(0) + 1) * c.hist_accel_step
            axis.plot(centres[ok], edge[ok], style, color=CATEGORICAL[1], linewidth=1.3, label=f"p{int(q * 100)} of a_tan")
        axis.axhline(0.0, color=INK["axis"], linewidth=0.8)
        axis.set_xlabel("speed (yd/s)")
        axis.set_title(f"{p} · {int(tot.sum()):,} frames", fontsize=10)
        axis.grid(False)
    ax[0][0].set_ylabel("tangential acceleration d|v|/dt (yd/s²)")
    ax[0][0].legend(frameon=False, fontsize=7, loc="lower left")
    finish(fig, "Unit: post-snap frame of an in-scope athlete (a frame is NOT an independent observation; the count is a density, not a "
                f"sample size). Colour: share of the frames of a {c.hist_speed_step:g} yd/s speed bin that fall in each "
                f"{c.hist_accel_step:g} yd/s² acceleration bin (bins with < {min_frames} frames are grey; accelerations beyond ± "
                f"{c.hist_accel_lim:g} yd/s² are in the end bins). Lines: median and 90th percentile of the acceleration at that speed "
                "(upper bin edge). This is the OBSERVED movement distribution of a position in its role, not a physical capability limit.",
           "Speed–acceleration phase space by position group")
    return fig


# ============================================================================== 05A · 6 starting-speed coverage
def plot_start_speed_coverage(plays: pl.DataFrame, c: OutcomeConfig):
    el = _eligible(plays)
    labs = [blab(lo, hi) for lo, hi in c.speed_bands]
    names = [f"{lo:g}–{hi:g}" for lo, hi in c.speed_bands]
    fig, ax = figure(1, 2)
    positions = _positions(el)
    if not positions:
        empty_panel(ax[0][0], "no eligible window")
        empty_panel(ax[0][1], "")
        finish(fig, "", "Starting-speed coverage")
        return fig
    w = {r["position_group"]: r for r in el.group_by("position_group").agg(
        pl.len().alias("n"), *[pl.col(f"band_acc_{b}").is_not_null().mean().alias(b) for b in labs]).iter_rows(named=True)}
    per = el.group_by("position_group", "nfl_id").agg(*[pl.col(f"band_acc_{b}").is_not_null().sum().alias(b) for b in labs])
    a = {r["position_group"]: r for r in per.group_by("position_group").agg(
        pl.len().alias("n"), *[(pl.col(b) >= c.primary_min_windows).mean().alias(b) for b in labs]).iter_rows(named=True)}
    _heat(ax[0][0], np.array([[_num(w[p][b]) for b in labs] for p in positions]), [f"{p} ({w[p]['n']:,} w)" for p in positions], names,
          fmt="{:.0%}")
    ax[0][0].set_title(f"Eligible windows with ≥ {c.min_band_starts} accelerating start frames in the band", fontsize=10)
    _heat(ax[0][1], np.array([[_num(a[p][b]) for b in labs] for p in positions]), [f"{p} ({a[p]['n']} ath)" for p in positions], names,
          fmt="{:.0%}")
    ax[0][1].set_title(f"Athletes with ≥ {c.primary_min_windows} such windows", fontsize=10)
    for axis in ax[0]:
        axis.set_xlabel("starting speed band (yd/s)")
    finish(fig, "Left, unit: eligible window (w); cell = share of the windows of the position on which the band-conditioned acceleration "
                "is defined. Right, unit: athlete (ath); cell = share of the athletes of the position who reach the primary exposure "
                "threshold in that band. A band that a position rarely reaches cannot support an athlete comparison there: this "
                "is a property of role and opportunity, and it decides which bands are feasible per position.",
           "Starting-speed coverage of the band-conditioned outcomes")
    return fig


# ============================================================================== 05A · 7 alignment
def plot_alignment(plays: pl.DataFrame, column: str, c: OutcomeConfig, top: int = 8):
    el = _eligible(plays).with_columns(pl.col("lined_up_position").fill_null("NA"))
    positions = _positions(el)
    fig, ax = figure(1, max(len(positions), 1))
    col = _pos_color(positions)
    if not positions:
        empty_panel(ax[0][0], "no eligible window")
    for axis, p in zip(ax[0], positions):
        t = (el.filter(pl.col("position_group") == p).group_by("lined_up_position")
             .agg(pl.len().alias("n"), pl.col("nfl_id").n_unique().alias("a"), pl.col(column).mean().alias("m"))
             .sort(["n", "lined_up_position"], descending=[True, False]).head(top))
        y = np.arange(t.height)
        axis.barh(y, t["n"].to_numpy(), color=col[p])
        for i, r in enumerate(t.iter_rows(named=True)):
            m = "–" if r["m"] is None else f"{r['m']:.2f}"
            axis.text(r["n"], i, f" {r['a']} ath · {m}", va="center", fontsize=7, color=INK["secondary"])
        axis.set_yticks(y)
        axis.set_yticklabels(t["lined_up_position"].to_list(), fontsize=8)
        axis.invert_yaxis()
        axis.set_xlim(0, float(t["n"].max()) * 1.7 if t.height else 1.0)
        axis.set_xlabel("eligible windows")
        axis.set_title(p, fontsize=10)
    finish(fig, f"Unit: eligible window. Bars: the {top} most frequent vendor alignments (`lined_up_position`) of each position group; "
                f"text: athletes observed in the alignment and the mean of the play measure «{measure_label(column)}» there. Alignment is "
                "an observed NFL label; the assignment of the athlete on the play is NOT in the data and is not inferred. Differences of "
                "the mean between alignments are differences of role, which an athlete comparison must not read as athlete quality.",
           "Movement opportunities by alignment")
    return fig


# ============================================================================== 05A · 8 game vs athlete
def plot_game_vs_athlete(games: pl.DataFrame, outcome_id: str, athletes: pl.DataFrame, c: OutcomeConfig, min_games: int = 4):
    g = (games.filter((pl.col("outcome_id") == outcome_id) & (pl.col("n_windows") >= c.min_game_windows))
         .join(athletes.select("nfl_id", "position_group"), on="nfl_id", how="inner"))
    per = (g.group_by("nfl_id").agg(pl.col("position_group").first(), pl.len().alias("k"), pl.col("mean").mean().alias("m"),
                                    pl.col("mean").min().alias("lo"), pl.col("mean").max().alias("hi"), pl.col("mean").std().alias("sd"))
           .filter(pl.col("k") >= min_games))
    positions = _positions(per)
    fig, ax = figure(1, max(len(positions), 1))
    col = _pos_color(positions)
    if not positions:
        empty_panel(ax[0][0], f"no athlete with ≥ {min_games} games")
    for axis, p in zip(ax[0], positions):
        t = per.filter(pl.col("position_group") == p).sort(["m", "nfl_id"])
        x = np.arange(t.height)
        axis.vlines(x, t["lo"].to_numpy(), t["hi"].to_numpy(), color=INK["axis"], linewidth=0.8)
        axis.plot(x, t["m"].to_numpy(), ".", color=col[p], markersize=4)
        between, within = float(t["m"].std() or float("nan")), float(t["sd"].median() or float("nan"))
        axis.set_title(f"{p} · {t.height} athletes\nSD between athletes {between:.2f} · median SD of an athlete's games {within:.2f}",
                       fontsize=9)
        axis.set_xlabel("athletes, sorted by their mean")
    ax[0][0].set_ylabel("athlete-game mean of the play measure")
    finish(fig, f"Outcome measure: {outcome_id}. Unit: athlete (dot = mean of the athlete's game means; line = range of the game "
                f"means). Athlete-games with ≥ {c.min_game_windows} windows, athletes with ≥ {min_games} such games. When the lines are long "
                "relative to the spread of the dots, a few games cannot rank athletes. The split-half reliability of notebook 05B "
                "quantifies this; the figure shows it.",
           "Within-athlete game variation against between-athlete spread")
    return fig


# ============================================================================== 05A · 9 smoothing
def variant_play_agreement(plays: pl.DataFrame, variants: pl.DataFrame, columns: Sequence[str], min_windows: int = 30) -> pl.DataFrame:
    """Play-level agreement of each variant with the primary measure on the SAME eligible windows (descriptive)."""
    schema = {"variant": pl.String, "measure": pl.String, "n_windows": pl.Int64, "pearson": pl.Float64, "mean_shift": pl.Float64,
              "shift_sd_units": pl.Float64, "sd_ratio": pl.Float64}
    base = _eligible(plays).select(*PP_KEYS, *columns)
    rows = []
    for v in (sorted(variants["variant"].unique().to_list()) if variants.height else []):
        other = variants.filter(pl.col("variant") == v).select(*PP_KEYS, *[pl.col(k).cast(pl.Float64).alias(f"{k}__v") for k in columns])
        j = base.join(other, on=PP_KEYS, how="inner")
        for k in columns:
            d = j.select(pl.col(k).cast(pl.Float64), f"{k}__v").drop_nulls()
            if d.height < min_windows:
                continue
            a, b = d[k].to_numpy(), d[f"{k}__v"].to_numpy()
            sd = float(a.std(ddof=1))
            r = float(np.corrcoef(a, b)[0, 1]) if sd > 0 and b.std() > 0 else float("nan")
            rows.append({"variant": v, "measure": k, "n_windows": d.height, "pearson": r, "mean_shift": float((b - a).mean()),
                         "shift_sd_units": float((b - a).mean() / sd) if sd > 0 else None,
                         "sd_ratio": float(b.std(ddof=1) / sd) if sd > 0 else None})
    return pl.DataFrame(rows, schema=schema)


def plot_smoothing_sensitivity(agreement: pl.DataFrame, c: OutcomeConfig):
    fig, ax = figure(1, 2)
    if not agreement.height:
        empty_panel(ax[0][0], "no variant computed")
        empty_panel(ax[0][1], "")
        finish(fig, "", "Smoothing and boundary sensitivity of the play measures")
        return fig
    measures = list(dict.fromkeys(agreement["measure"].to_list()))
    variants = sorted(agreement["variant"].unique().to_list())
    got = {(r["measure"], r["variant"]): r for r in agreement.iter_rows(named=True)}
    R = np.array([[_num(got.get((m, v), {}).get("pearson")) for v in variants] for m in measures])
    D = np.array([[_num(got.get((m, v), {}).get("shift_sd_units")) for v in variants] for m in measures])
    _heat(ax[0][0], R, measures, variants, colors=[STATUS["critical"], "#ffffff", SEQUENTIAL_BLUE[4]], vmin=0.0, vmax=1.0)
    ax[0][0].set_title("Pearson correlation with the primary measure (same windows)", fontsize=10)
    _heat(ax[0][1], D, measures, variants, fmt="{:+.2f}", colors=[CATEGORICAL[1], "#ffffff", CATEGORICAL[0]], vmin=-1.0, vmax=1.0)
    ax[0][1].set_title("Mean shift, variant − primary (SD of the primary measure)", fontsize=10)
    n = int(agreement["n_windows"].max())
    finish(fig, f"Unit: eligible window (up to {n:,} per cell; every {max(1, c.sensitivity_game_stride)}th game of a batch). Columns: "
                "sg<window>p<order> = alternative Savitzky–Golay filter; boundary_incl = primary filter with windows touching "
                "filter-edge frames kept. Primary: "
                f"sg{c.primary_smoothing[0]}p{c.primary_smoothing[1]}, edge windows excluded. A measure whose level moves with the filter "
                "is a property of the filter as much as of the athlete; athlete-level rank agreement is tested in notebook 05B.",
           "Smoothing and boundary sensitivity of the play measures")
    return fig


# ============================================================================== 05A · 10 exclusions and missingness
def plot_exclusions_missingness(plays: pl.DataFrame, columns: Sequence[str], c: OutcomeConfig):
    fig, ax = figure(1, 2)
    a, b = ax[0]
    d = plays.with_columns(pl.col("position_group").fill_null("UNKNOWN"))
    groups = sorted(d["position_group"].unique().to_list())
    t = {(r["position_group"], r["exclusion_reason"]): r["len"] for r in d.group_by("position_group", "exclusion_reason").len().iter_rows(named=True)}
    tot = {g: sum(v for (gg, _), v in t.items() if gg == g) for g in groups}
    left = np.zeros(len(groups))
    for reason, colr in REASON_COLOR.items():
        v = np.array([t.get((g, reason), 0) / max(tot[g], 1) for g in groups])
        a.barh(np.arange(len(groups)), v, left=left, color=colr, label=reason.replace("_", " ") or "eligible")
        left += v
    a.set_yticks(np.arange(len(groups)))
    a.set_yticklabels([f"{g} ({tot[g]:,})" for g in groups], fontsize=9)
    a.invert_yaxis()
    a.set_xlim(0, 1)
    a.set_xlabel("share of tracked player-plays with a snap tag")
    a.legend(frameon=False, fontsize=7, loc="lower right")
    a.set_title("First failing eligibility rule", fontsize=11)
    el = _eligible(plays)
    positions = _positions(el)
    if positions:
        m = {r["position_group"]: r for r in el.group_by("position_group").agg(
            *[pl.col(k).is_null().mean().alias(k) for k in columns]).iter_rows(named=True)}
        _heat(b, np.array([[_num(m[p][k]) for p in positions] for k in columns]), list(columns), positions, fmt="{:.0%}",
              colors=["#ffffff", CATEGORICAL[1]])
    else:
        empty_panel(b, "no eligible window")
    b.set_title("Eligible windows on which the measure is NOT defined", fontsize=11)
    finish(fig, "Left, unit: tracked player-play (count in brackets); rules are applied in this order — position in scope, not nullified "
                "by penalty, quality record present, usable quality tier, enough post-snap frames — and the first failing rule is the "
                "reason. Right, unit: eligible window; a measure is undefined when no fixed window fits inside a "
                "tracking segment away from filter-edge frames, when a speed band is not visited, or when no braking qualifies. An "
                "undefined measure is excluded from the athlete statistic — never filled.",
           "Exclusions and measure missingness")
    return fig


# ============================================================================== 05A · 11 examples
def plot_examples(items: Sequence[dict[str, Any]], ncols: int, c: OutcomeConfig):
    """``items``: dicts with ``frames`` (``nfl_outcomes.start_table`` rows of ONE recorded player-play) and ``title``."""
    n = max(len(items), 1)
    nrows = int(math.ceil(n / max(ncols, 1)))
    fig, ax = figure(nrows, ncols, size=(12.8, max(7.2, 2.6 * nrows + 1.2)))
    flat = [x for row in ax for x in row]
    lb, w0 = wlab(c.sustain_s[0]), float(c.sustain_s[0])
    for axis, it in zip(flat, items):
        f = it["frames"].sort("t_rel")
        if not f.height:
            empty_panel(axis, "no recorded frame")
            continue
        t, v = f["t_rel"].to_numpy(), f["v0"].to_numpy()
        a = f["a0"].cast(pl.Float64).fill_null(float("nan")).to_numpy()
        axis.plot(t, v, color=CATEGORICAL[0], linewidth=1.6, label="speed")
        edge = f["boundary"].to_numpy()
        if edge.any():
            axis.plot(t[edge], v[edge], "x", color=INK["muted"], markersize=4, label="filter-edge frame")
        g = f[f"gain{lb}"].cast(pl.Float64).fill_null(float("nan")).to_numpy()
        if np.isfinite(g).any():
            i = int(np.nanargmax(g))
            axis.axvspan(t[i], t[i] + w0, color=CATEGORICAL[3], alpha=0.18, label=f"largest {w0:g} s gain")
        twin = axis.twinx()
        twin.plot(t, a, color=CATEGORICAL[1], linewidth=0.9, alpha=0.8)
        twin.axhline(0.0, color=INK["axis"], linewidth=0.6)
        twin.set_ylabel("a_tan (yd/s²)", fontsize=7, color=CATEGORICAL[1])
        twin.tick_params(labelsize=7)
        twin.grid(False)
        axis.set_title(it["title"], fontsize=8)
        axis.set_xlabel("seconds after the snap", fontsize=8)
        axis.set_ylabel("speed (yd/s)", fontsize=8)
    for axis in flat[len(items):]:
        empty_panel(axis, "")
    if items:
        flat[0].legend(frameon=False, fontsize=7, loc="upper left")
    finish(fig, "Each panel is ONE recorded player-play (real tracking, recomputed from the silver frames with the primary filter). "
                "Blue: derived speed; orange: tangential acceleration d|v|/dt (right axis); crosses: frames inside the filter boundary "
                f"of their tracking segment; shaded: the {w0:g} s window with the largest speed gain that is not a boundary window. Examples "
                "are chosen by rule (the window closest to a stated quantile of its position group), not by hand.",
           "Example recorded movement sequences")
    return fig


# ============================================================================== 05A · 12 exposure vs opportunity
def plot_exposure_vs_opportunity(outcomes: pl.DataFrame, coverage: pl.DataFrame, base_id: str, cond_id: str, c: OutcomeConfig):
    fig, ax = figure(1, 2)
    a, b = ax[0]
    base = outcomes.filter(pl.col("outcome_id") == base_id).select("nfl_id", "position_group", pl.col("n_windows").alias("n_all"))
    j = base.join(outcomes.filter(pl.col("outcome_id") == cond_id).select("nfl_id", pl.col("n_windows").alias("n_cond")),
                  on="nfl_id", how="left").with_columns(pl.col("n_cond").fill_null(0))
    positions = _positions(j)
    col = _pos_color(positions)
    for p in positions:
        s = j.filter(pl.col("position_group") == p)
        a.plot(s["n_all"].to_numpy(), s["n_cond"].to_numpy(), "o", color=col[p], markersize=3.5, alpha=0.7, label=f"{p} ({s.height})")
    if positions:
        hi = float(j["n_all"].max())
        a.plot([1, hi], [1, hi], color=INK["axis"], linewidth=0.8)
        a.axhline(c.primary_min_windows, color=INK["muted"], linewidth=0.8, linestyle="--")
        a.set_xscale("log")
        a.set_yscale("symlog", linthresh=1)
        a.legend(frameon=False, fontsize=8)
    else:
        empty_panel(a, "no athlete")
    a.set_xlabel(f"eligible windows with {_short(base_id)} defined (tracked exposure)")
    a.set_ylabel(f"eligible windows with {_short(cond_id)} defined (observed opportunity)")
    a.set_title("Tracked exposure against observed opportunity, per athlete", fontsize=10)
    cov = coverage.filter((pl.col("position_group") != ALL) & pl.col("opportunity_rate").is_not_null())
    ids = list(dict.fromkeys(cov.sort("outcome_id")["outcome_id"].to_list()))
    pos = _positions(cov)
    if ids and pos:
        got = {(r["outcome_id"], r["position_group"]): r["opportunity_rate"] for r in cov.iter_rows(named=True)}
        _heat(b, np.array([[_num(got.get((o, p))) for p in pos] for o in ids]), [_short(o) for o in ids], pos, fmt="{:.0%}", fontsize=7)
    else:
        empty_panel(b, "no coverage row")
    b.set_title("Opportunity rate: windows with the outcome defined / eligible windows", fontsize=10)
    finish(fig, "Left, unit: athlete (count per position in the legend); dashed line: the primary exposure threshold "
                f"({c.primary_min_windows} windows). An athlete on the diagonal has the conditional outcome on every eligible window; "
                "athletes far below it were tracked but rarely in the situation the outcome needs. Right, unit: eligible window; "
                "composites have no rate. Frequency of opportunity (this figure) is kept apart from movement conditional on an "
                "opportunity (the outcome values).",
           "Tracked exposure against observed movement opportunities")
    return fig


# ============================================================================== 05B · 1 reliability
def plot_reliability_forest(reliability: pl.DataFrame, screen: pl.DataFrame, c: OutcomeConfig):
    rows = screen.sort("family", "outcome_id").to_dicts()
    fig, ax = figure(1, 1, size=(12.8, max(7.2, 0.32 * len(rows) + 2.4)))
    a = ax[0][0]
    if not rows:
        empty_panel(a, "no outcome screened")
    for i, r in enumerate(rows):
        colr = FAMILY_COLOR.get(r["family"], INK["muted"])
        oe = _pick(reliability, r["outcome_id"], "game_odd_even_sb", c)
        rh = _pick(reliability, r["outcome_id"], "game_random_halves_sb", c)
        adj = _pick(reliability, r["outcome_id"], "game_random_halves_sb", c, adj="context")
        if oe and oe["estimate"] is not None:
            if oe["ci_lower"] is not None and oe["ci_upper"] is not None:
                a.plot([oe["ci_lower"], oe["ci_upper"]], [i, i], color=colr, linewidth=1.6)
            a.plot([oe["estimate"]], [i], "o", color=colr, markersize=5)
        if rh and rh["estimate"] is not None:
            a.plot([rh["estimate"]], [i], "D", color=INK["primary"], markersize=4)
        if adj and adj["estimate"] is not None:
            a.plot([adj["estimate"]], [i], "s", markerfacecolor="none", markeredgecolor=INK["secondary"], markersize=6)
        a.text(1.02, i, f"n = {r['n_athletes']} · {r['status']}", va="center", fontsize=7, color=STATUS_COLOR.get(r["status"], INK["muted"]),
               transform=a.get_yaxis_transform())
    a.axvline(c.ref_reliability, color=INK["muted"], linewidth=0.8, linestyle="--")
    a.axvline(c.ref_reliability_lower, color=INK["axis"], linewidth=0.8, linestyle=":")
    a.set_yticks(np.arange(len(rows)))
    a.set_yticklabels([r["outcome_id"] for r in rows], fontsize=8)
    a.invert_yaxis()
    a.set_xlim(-0.2, 1.0)
    a.set_xlabel("reliability of the athlete value across independent games (stepped up to all games)")
    from matplotlib.lines import Line2D

    a.legend(handles=[Line2D([], [], marker="o", color=INK["secondary"], label="odd / even games, 95 % athlete bootstrap"),
                      Line2D([], [], marker="D", color=INK["primary"], linestyle="", label="mean of random game half-splits"),
                      Line2D([], [], marker="s", markerfacecolor="none", color=INK["secondary"], linestyle="", label="same, context-adjusted")],
             frameon=False, fontsize=8, loc="lower left")
    finish(fig, f"Unit: athlete with ≥ {c.primary_min_windows} eligible windows (n at the right), position-group means removed. Halves are "
                "sets of whole GAMES of the athlete, never frames or plays; the interval resamples athletes. Reference lines "
                f"{c.ref_reliability:g} (dashed) and {c.ref_reliability_lower:g} for the lower bound (dotted) are heuristics. Colour: "
                "outcome family O1–O6. Reliability shows that a value repeats for an athlete; role and usage repeat too.",
           "Reliability of the candidate NFL movement outcomes across games")
    return fig


def plot_reliability_by_position(reliability: pl.DataFrame, c: OutcomeConfig, method: str = "game_random_halves_sb"):
    r = reliability.filter((pl.col("method") == method) & (pl.col("adjustment") == "raw") & (pl.col("min_windows") == c.primary_min_windows))
    fig, ax = figure(1, 1)
    a = ax[0][0]
    ids = sorted(r["outcome_id"].unique().to_list())
    groups = [WITHIN, *_positions(r, "group")]
    if not ids:
        empty_panel(a, "no reliability row at the primary threshold")
    else:
        got = {(x["outcome_id"], x["group"]): x for x in r.iter_rows(named=True)}
        M = np.array([[_num(got.get((o, g), {}).get("estimate")) for g in groups] for o in ids])
        lab = np.array([[f"{got[(o, g)]['estimate']:.2f} ({got[(o, g)]['n_athletes']})" if (o, g) in got and got[(o, g)]["estimate"] is not None
                         else "" for g in groups] for o in ids], dtype=object)
        _heat(a, M, ids, groups, colors=[STATUS["critical"], "#ffffff", SEQUENTIAL_BLUE[4]], vmin=0.0, vmax=1.0, labels=lab)
    finish(fig, f"Unit: athlete with ≥ {c.primary_min_windows} eligible windows; cell = mean stepped-up correlation over random half-splits "
                f"of each athlete's games, athletes in brackets. Column {WITHIN}: all positions with their means removed. A position "
                f"with fewer than {c.min_athletes_position} athletes has no estimate (grey). Small groups give unstable values: read the "
                "count before the colour.",
           "Reliability across games inside each position group")
    return fig


# ============================================================================== 05B · 2 exposure
def plot_exposure_curves(exposure: pl.DataFrame, outcome_ids: Sequence[str], c: OutcomeConfig, cohort: str = "available"):
    fig, ax = figure(1, 2)
    for axis, kind in zip(ax[0], ("windows", "games")):
        drawn = False
        for k, oid in enumerate(outcome_ids):
            colr = CATEGORICAL[k % len(CATEGORICAL)]
            e = exposure.filter((pl.col("outcome_id") == oid) & (pl.col("budget_type") == kind))
            s = e.filter((pl.col("method") == "first_vs_later_games") & (pl.col("cohort") == cohort) & pl.col("estimate").is_not_null()).sort("budget")
            if s.height:
                x, y = s["budget"].to_numpy().astype(float), s["estimate"].to_numpy()
                lo = np.array([_num(v) for v in s["ci_lower"].to_list()])
                hi = np.array([_num(v) for v in s["ci_upper"].to_list()])
                axis.plot(x, y, "-o", color=colr, markersize=4, label=_short(oid))
                ok = np.isfinite(lo) & np.isfinite(hi)
                axis.vlines(x[ok], lo[ok], hi[ok], color=colr, linewidth=1.0)
                for xi, yi, ni in zip(x, y, s["n_athletes"].to_list()):
                    axis.text(xi, yi, f" {ni}", fontsize=6, color=colr, va="bottom")
                drawn = True
            m = e.filter((pl.col("method") == "model_projection") & pl.col("estimate").is_not_null()).sort("budget")
            if m.height:
                axis.plot(m["budget"].to_numpy().astype(float), m["estimate"].to_numpy(), "--", color=colr, linewidth=0.9)
        if drawn:
            axis.set_xscale("log")
            axis.legend(frameon=False, fontsize=7)
        else:
            empty_panel(axis, f"no athlete with a later-games reference at any {kind} budget")
        axis.axhline(c.ref_reliability, color=INK["muted"], linewidth=0.8, linestyle="--")
        axis.set_ylim(-0.2, 1.0)
        axis.set_xlabel(f"first {kind} of the athlete, in time order (log axis)")
        axis.set_ylabel("Pearson with the outcome on LATER games (within position)")
        axis.set_title(f"Observation budget in {kind}", fontsize=11)
    finish(fig, f"Unit: athlete (count beside each point) with the budget and ≥ {c.holdout_min_windows} windows in games played strictly after "
                "it. Solid: correlation between the outcome on the first N windows (or G games) and on the later games, 95 % athlete "
                "bootstrap — attenuated by the noise of BOTH sides, so a lower bound of the reliability of the first-N value. Dashed "
                "(windows panel): projection from the variance components of the play measure, which describes an athlete MEAN. High "
                "reliability on all windows says nothing about the first 20.",
           "Reliability against observation budget")
    return fig


def plot_variance_components(reliability: pl.DataFrame, c: OutcomeConfig):
    r = reliability.filter((pl.col("adjustment") == "raw") & pl.col("method").is_in(["var_athlete", "var_game", "var_window", "play_icc1", "game_icc1"]))
    ids = sorted(r["outcome_id"].unique().to_list())
    fig, ax = figure(1, 1, size=(12.8, max(7.2, 0.32 * len(ids) + 2.4)))
    a = ax[0][0]
    if not ids:
        empty_panel(a, "no variance component")
    got = {(x["outcome_id"], x["method"]): _num(x["estimate"]) for x in r.iter_rows(named=True)}
    parts = (("var_athlete", "between athletes", SEQUENTIAL_BLUE[5]), ("var_game", "between games of an athlete", SEQUENTIAL_BLUE[2]),
             ("var_window", "between windows of a game", INK["grid"]))
    for i, o in enumerate(ids):
        v = np.array([got.get((o, k), float("nan")) for k, _, _ in parts])
        tot = np.nansum(v)
        left = 0.0
        for (k, lab, colr), x in zip(parts, v):
            share = x / tot if tot > 0 and np.isfinite(x) else 0.0
            a.barh(i, share, left=left, color=colr, label=lab if i == 0 else None)
            left += share
        a.text(1.01, i, f"window ICC {got.get((o, 'play_icc1'), float('nan')):.2f} · game-mean ICC {got.get((o, 'game_icc1'), float('nan')):.2f}",
               va="center", fontsize=7, color=INK["secondary"])
    a.set_yticks(np.arange(len(ids)))
    a.set_yticklabels(ids, fontsize=8)
    a.invert_yaxis()
    a.set_xlim(0, 1.45)
    a.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    a.set_xlabel("share of the variance of the play measure (within position)")
    if ids:
        a.legend(frameon=False, fontsize=8, loc="lower right")
    finish(fig, f"Unit: eligible window of athletes with ≥ {c.primary_min_windows} windows; one-way moment estimators (approximate for "
                "unbalanced data). The athlete share is what a single window tells about the athlete; the game share is noise that more "
                "windows of the same game cannot remove — only more games can. Composites have no play measure and are absent.",
           "Where the variance of each play measure lies: athlete, game, window")
    return fig


# ============================================================================== 05B · 3 context, sensitivity, redundancy
def plot_context(context: pl.DataFrame, c: OutcomeConfig):
    t = context.filter((pl.col("level") == "__ALL__") & pl.col("eta2").is_not_null())
    fig, ax = figure(1, 1)
    a = ax[0][0]
    ids = sorted(t["outcome_id"].unique().to_list())
    variables = [v for v in ("lined_up_position", "play_class", "side", "down", "context_cell") if v in set(t["variable"].to_list())]
    if not ids or not variables:
        empty_panel(a, "no context row")
    else:
        mx = {(r["outcome_id"], r["variable"]): r["eta2"] for r in t.group_by("outcome_id", "variable").agg(pl.col("eta2").max()).iter_rows(named=True)}
        _heat(a, np.array([[_num(mx.get((o, v))) for v in variables] for o in ids]), ids, variables, colors=["#ffffff", CATEGORICAL[1]],
              vmin=0.0, vmax=0.5)
    finish(fig, "Unit: eligible window. Cell = largest value over position groups of the between-level share (η²) of the play-measure "
                "variance explained by one observed context variable; context_cell = alignment × play class, the cell of the adjusted "
                f"outcome. Reference line of the screening: {c.ref_context_share:g}. Only observed labels are used; assignment, opponent "
                "and scheme are not in the data, so a low share does not prove that context is controlled.",
           "Share of the play-measure variance explained by observed context")
    return fig


def plot_sensitivity(sensitivity: pl.DataFrame, c: OutcomeConfig):
    fig, ax = figure(1, 2)
    if not sensitivity.height:
        empty_panel(ax[0][0], "no sensitivity row")
        empty_panel(ax[0][1], "")
        finish(fig, "", "Sensitivity of the athlete-level outcomes")
        return fig
    ids = sorted(sensitivity["outcome_id"].unique().to_list())
    variants = sorted(sensitivity["variant"].unique().to_list())
    got = {(r["outcome_id"], r["variant"]): r for r in sensitivity.iter_rows(named=True)}
    R = np.array([[_num(got.get((o, v), {}).get("spearman")) for v in variants] for o in ids])
    D = np.array([[_num(got.get((o, v), {}).get("shift_sd_units")) for v in variants] for o in ids])
    _heat(ax[0][0], R, ids, variants, colors=[STATUS["critical"], "#ffffff", SEQUENTIAL_BLUE[4]], vmin=0.5, vmax=1.0, fontsize=7)
    ax[0][0].set_title("Rank agreement with the primary definition (Spearman, within position)", fontsize=10)
    _heat(ax[0][1], D, ids, variants, fmt="{:+.2f}", colors=[CATEGORICAL[1], "#ffffff", CATEGORICAL[0]], vmin=-1.0, vmax=1.0, fontsize=7)
    ax[0][1].set_yticklabels([])
    ax[0][1].set_title("Mean shift, variant − primary (SD of the primary athlete values)", fontsize=10)
    for axis in ax[0]:
        axis.tick_params(axis="x", labelrotation=30)
    finish(fig, "Unit: athlete. sg<window>p<order> and boundary_incl: the same windows under another filter or with filter-edge windows "
                "kept (subset of games). tier_A_only, in_player_play_only, post_snap_ge_3s, boundary_frac_le_0p2: the primary filter on "
                f"the windows passing a quality rule. Reference line for the filter variants: rank agreement ≥ {c.ref_smoothing_rank:g}.",
           "Sensitivity of the athlete-level outcomes to smoothing, boundaries and quality filters")
    return fig


def plot_redundancy(redundancy: pl.DataFrame, ids: Sequence[str], c: OutcomeConfig):
    fig, ax = figure(1, 1, size="square")
    a = ax[0][0]
    ids = list(ids)
    if not redundancy.height or len(ids) < 2:
        empty_panel(a, "fewer than two outcomes to compare")
    else:
        got = {}
        for r in redundancy.iter_rows(named=True):
            got[(r["outcome_a"], r["outcome_b"])] = got[(r["outcome_b"], r["outcome_a"])] = _num(r["spearman"])
        M = np.array([[1.0 if x == y else got.get((x, y), float("nan")) for y in ids] for x in ids])
        _heat(a, M, [_short(i) for i in ids], [_short(i) for i in ids], colors=[CATEGORICAL[1], "#ffffff", SEQUENTIAL_BLUE[4]],
              vmin=-1.0, vmax=1.0, fontsize=6)
        a.tick_params(axis="x", labelrotation=90)
    finish(fig, f"Unit: athlete with ≥ {c.primary_min_windows} windows on both outcomes; Spearman after removing position-group means. "
                f"Pairs above {c.ref_redundancy:g} rank athletes almost identically and are not counted as separate targets.",
           "Redundancy between NFL movement outcomes")
    return fig


# ============================================================================== 05B · 4 screen and freeze evidence
def plot_screen(screen: pl.DataFrame, c: OutcomeConfig):
    rows = screen.sort("family", "outcome_id").to_dicts()
    flags = ("pass_coverage", "pass_reliability", "pass_smoothing", "pass_exposure", "pass_context")
    fig, ax = figure(1, 1, size=(12.8, max(7.2, 0.32 * len(rows) + 2.4)))
    a = ax[0][0]
    if not rows:
        empty_panel(a, "no outcome screened")
    else:
        M = np.array([[1.0 if r[f] else 0.0 for f in flags] for r in rows])
        vals = (("n_athletes", "{:.0f}"), ("reliability", "{:.2f}"), ("smoothing_min_spearman", "{:.2f}"), ("exposure_rho", "{:+.2f}"),
                ("context_eta2", "{:.2f}"))
        lab = np.array([["–" if r[k] is None else fmt.format(r[k]) for k, fmt in vals] for r in rows], dtype=object)
        _heat(a, M, [r["outcome_id"] for r in rows], ["coverage\n(athletes)", "reliability\n(random halves)", "smoothing\n(min rank agr.)",
                                                      "exposure\n(ρ with log windows)", "context\n(max η²)"],
              colors=["#f6d5d5", "#d5efd5"], vmin=0.0, vmax=1.0, labels=lab)
        for i, r in enumerate(rows):
            a.text(len(flags) - 0.4, i, f"{r['status']} ({r['role']})", va="center", fontsize=8, color=STATUS_COLOR.get(r["status"], INK["muted"]))
        a.set_xlim(-0.5, len(flags) + 1.6)
    finish(fig, "Unit: candidate outcome. Green = the reference line is met, red = not met; the number is the evidence. Status: eligible = "
                "all five met; provisional = coverage and reliability met; reference, component and exploratory outcomes are never "
                "targets whatever their numbers. The lines are documented heuristics and the labels are for review: the freeze is "
                "recorded by the analyst with a rationale, and no Combine association enters it.",
           "Screening of the candidate NFL movement outcomes")
    return fig


def plot_first_vs_later(pairs: pl.DataFrame, outcome_id: str, budget: int, kind: str, unit: str, c: OutcomeConfig):
    fig, ax = figure(1, 1, size="square")
    a = ax[0][0]
    p = pairs.filter(pl.col("n_1") >= c.holdout_min_windows).with_columns(
        (pl.col("value_0") - pl.col("value_0").mean().over("position_group")).alias("x"),
        (pl.col("value_1") - pl.col("value_1").mean().over("position_group")).alias("y"))
    positions = _positions(p)
    col = _pos_color(positions)
    if not positions:
        empty_panel(a, "no athlete with the budget and a later-games reference")
    else:
        for g in positions:
            s = p.filter(pl.col("position_group") == g)
            a.plot(s["x"].to_numpy(), s["y"].to_numpy(), "o", color=col[g], markersize=4, alpha=0.75, label=f"{g} ({s.height})")
        x, y = p["x"].to_numpy(), p["y"].to_numpy()
        lim = float(np.nanmax(np.abs(np.r_[x, y]))) * 1.1 or 1.0
        a.plot([-lim, lim], [-lim, lim], color=INK["axis"], linewidth=0.8)
        a.set_xlim(-lim, lim)
        a.set_ylim(-lim, lim)
        r = float(np.corrcoef(x, y)[0, 1]) if p.height > 2 and x.std() > 0 and y.std() > 0 else float("nan")
        a.set_title(f"{outcome_id} · Pearson {r:.2f} · {p.height} athletes", fontsize=11)
        a.legend(frameon=False, fontsize=8)
    a.set_xlabel(f"first {budget} {kind}: value − position mean ({unit})")
    a.set_ylabel(f"later games: value − position mean ({unit})")
    finish(fig, f"Unit: athlete with {budget} {kind} and ≥ {c.holdout_min_windows} windows in games played strictly after them. The two "
                "axes never share a game. Both are noisy, so the correlation understates the reliability of either side.",
           "Early exposure against later games")
    return fig


def plot_season_stability(reliability: pl.DataFrame, c: OutcomeConfig):
    r = reliability.filter(pl.col("method").str.starts_with("season_") & (pl.col("adjustment") == "raw") & pl.col("estimate").is_not_null())
    methods = sorted(r["method"].unique().to_list())
    fig, ax = figure(1, max(len(methods), 1))
    if not methods:
        empty_panel(ax[0][0], f"no pair of consecutive seasons with ≥ {c.min_athletes} athletes")
    for axis, m in zip(ax[0], methods):
        rows = r.filter(pl.col("method") == m).sort("outcome_id").to_dicts()
        for i, x in enumerate(rows):
            colr = FAMILY_COLOR.get(x["outcome_id"][:2], INK["muted"])
            if x["ci_lower"] is not None and x["ci_upper"] is not None:
                axis.plot([x["ci_lower"], x["ci_upper"]], [i, i], color=colr, linewidth=1.5)
            axis.plot([x["estimate"]], [i], "o", color=colr, markersize=5)
            axis.text(1.02, i, f"n = {x['n_athletes']}", va="center", fontsize=7, color=INK["secondary"], transform=axis.get_yaxis_transform())
        axis.set_yticks(np.arange(len(rows)))
        axis.set_yticklabels([x["outcome_id"] for x in rows], fontsize=8)
        axis.invert_yaxis()
        axis.set_xlim(-0.3, 1.0)
        axis.axvline(0.0, color=INK["axis"], linewidth=0.8)
        axis.set_xlabel("Pearson between seasons (within position)")
        axis.set_title(m.replace("season_", "").replace("_to_", " → "), fontsize=11)
    finish(fig, "Unit: athlete with enough windows in BOTH seasons (n at the right); 95 % athlete bootstrap; not stepped up. A season "
                "pair mixes measurement repeatability with real change of role, usage and development between seasons, and "
                "the athletes observed twice are a selected group.",
           "Stability of the outcomes between consecutive seasons")
    return fig
