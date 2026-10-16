"""Movement-fingerprint similarity from the frozen TCN-B representation. NumPy / Polars only — no encoder is trained.

Source          the fold-fitted attempt embeddings persisted by notebook 06B (``transfer_attempt_representations_outer``,
                family TCN). Axes of different fold encoders are NOT the same axes, so every comparison is made inside
                ONE encoder: an athlete is compared with the others under the encoder of the fold in which HE was held
                out (inductive: that encoder never saw his runs). An athlete never held out is compared under the first
                fold encoder and flagged ``inductive = False``.
Distance        Euclidean distance between athlete embeddings (mean of valid attempts) after scaling each dimension by
                its SD among the encoder's training athletes, divided by √dimensions.
What is shown   WHO the athlete moves like (nearest athletes of his position; nearest among athletes with near-identical
                official times) and HOW STABLE that is (his two attempts against other athletes; training seeds). A
                latent coordinate has no physical name and being far from everybody is not a quality: there is no
                "explosiveness", "uniqueness" or "NFL readiness" value here.
"""

from __future__ import annotations

from typing import Any, Callable, Sequence

import numpy as np
import polars as pl

from ...contracts import TableContract

S, I, F, B = pl.String(), pl.Int64(), pl.Float64(), pl.Boolean()
NAN = float("nan")

NEIGHBORS = TableContract(
    "combine_fingerprint_neighbors", "Nearest athletes in the TCN-B representation, within position.",
    {"nfl_id": I, "kind": S, "rank": I, "neighbor_id": I, "position_group": S, "fold": I, "inductive": B, "distance": F,
     "distance_pct": F, "d_forty": F, "d_ten": F, "n_candidates": I}, key=("nfl_id", "kind", "rank"), kind="dataset",
    area="athlete × neighbour kind × rank",
    docs={"kind": "position = nearest athletes of the position · similar_official = nearest among athletes within the "
                  "official-time caliper (same position)",
          "distance_pct": "share of the position's athletes that are CLOSER to the query than this neighbour (0 = nearest)",
          "inductive": "the encoder used never saw the query athlete's runs", "n_candidates": "athletes the neighbour was chosen from"})
STABILITY = TableContract(
    "combine_fingerprint_stability", "Repeat-attempt and training-seed stability of each athlete's fingerprint.",
    {"nfl_id": I, "position_group": S, "fold": I, "inductive": B, "n_attempts": I, "self_distance": F, "self_auc": F,
     "nearest_other_distance": F, "seed_overlap": F, "random_overlap": F, "first_speed_gap": F, "start_gap_s": F},
    key=("nfl_id",), area="athlete",
    docs={"self_distance": "distance between the athlete's first two attempts",
          "self_auc": "share of other athletes of the position whose first attempt lies FARTHER from his first attempt than his own "
                      "second attempt (1 = his repeat is closer than everybody; 0.5 = not distinguishable)",
          "seed_overlap": "mean Jaccard overlap of his k nearest neighbours between the primary encoder and each seed replicate",
          "random_overlap": "same overlap against an UNTRAINED encoder (what a trivial representation already gives)",
          "first_speed_gap": "difference of first-frame speed between the two attempts [yd/s] (recording, not athlete)",
          "start_gap_s": "difference of the recording start between the two attempts [s]"})


# ============================================================================== embeddings of one encoder
class FoldEmbeddings:
    """Attempt and athlete embeddings of ONE (scheme, family, fold) encoder."""

    def __init__(self, g: pl.DataFrame, order: dict[str, int]):
        g = g.sort("seq_id", "dim")
        E = g["dim"].n_unique()
        if g.height == 0 or g.height % E:
            raise ValueError("the attempt table is empty or not a whole number of embeddings")
        pick = slice(None, None, E)
        self.seq = g["seq_id"].to_numpy()[pick].astype(str)
        self.nfl = g["nfl_id"].to_numpy()[pick].astype(np.int64)
        self.train = g["role"].to_numpy()[pick].astype(str) == "train"
        Z = g["value"].to_numpy().reshape(-1, E).astype(float)
        self.order = np.array([order.get(s, 10 ** 6) for s in self.seq])
        ids, inv = np.unique(self.nfl, return_inverse=True)
        n = np.bincount(inv).astype(float)
        mean = np.column_stack([np.bincount(inv, weights=Z[:, j]) for j in range(E)]) / n[:, None]
        tr_a = np.bincount(inv, weights=self.train.astype(float)) > 0
        base = mean[tr_a] if tr_a.sum() >= 3 else mean
        self.scale = np.where(base.std(0) > 1e-12, base.std(0), 1.0)
        self.ids, self.n, self.E, self.heldout = ids, n.astype(np.int64), E, ~tr_a
        self.Z = mean / self.scale
        self.attempt = Z / self.scale
        self._inv = inv

    def attempts_of(self, i: int) -> np.ndarray:
        """Scaled embeddings of athlete index ``i``, first attempt first."""
        k = np.flatnonzero(self._inv == i)
        return self.attempt[k[np.argsort(self.order[k], kind="stable")]]

    def distances(self, i: int, others: np.ndarray) -> np.ndarray:
        return np.sqrt(((self.Z[others] - self.Z[i]) ** 2).sum(1) / self.E)


def fold_embeddings(attempts: pl.DataFrame, order: dict[str, int], family: str = "TCN", scheme: str = "outer") -> dict[int, FoldEmbeddings]:
    t = attempts.filter((pl.col("scheme") == scheme) & (pl.col("family") == family))
    return {int(f): FoldEmbeddings(g, order) for (f,), g in sorted(t.partition_by(["fold"], as_dict=True).items())}


def query_fold(emb: dict[int, FoldEmbeddings]) -> dict[int, tuple[int, bool]]:
    """athlete id → (fold whose encoder is used for him, inductive?)."""
    out: dict[int, tuple[int, bool]] = {}
    first = min(emb)
    for f in sorted(emb):
        e = emb[f]
        for i in e.ids[e.heldout].tolist():
            out.setdefault(int(i), (f, True))
    for i in emb[first].ids.tolist():
        out.setdefault(int(i), (first, False))
    return out


# ============================================================================== neighbours
def neighbor_table(emb: dict[int, FoldEmbeddings], position: dict[int, str], official: dict[int, tuple[float, float]] | None = None,
                   k: int = 5, caliper: tuple[float, float] = (0.05, 0.03), queries: Sequence[int] | None = None) -> pl.DataFrame:
    """k nearest athletes of the same position for every query athlete (kind 'position'), and — when official forty and
    10-yard times are given — the k nearest among athletes within ``caliper`` (|Δ forty| s, |Δ 10-yard| s) of the
    query (kind 'similar_official'). Candidates are chosen by position and official times only."""
    qf = query_fold(emb)
    rows: list[dict[str, Any]] = []
    for q in (sorted(qf) if queries is None else [int(x) for x in queries if int(x) in qf]):
        f, ind = qf[q]
        e = emb[f]
        i = int(np.searchsorted(e.ids, q))
        pos = position.get(q)
        cand = np.array([j for j in range(e.ids.size) if j != i and position.get(int(e.ids[j])) == pos], dtype=np.int64)
        if pos is None or cand.size == 0:
            continue
        d = e.distances(i, cand)
        o = np.argsort(d, kind="stable")
        off = official or {}
        fq = off.get(q, (NAN, NAN))
        df = np.array([off.get(int(e.ids[j]), (NAN, NAN))[0] - fq[0] for j in cand])
        dt = np.array([off.get(int(e.ids[j]), (NAN, NAN))[1] - fq[1] for j in cand])
        plans = [("position", o[:k], cand.size)]
        if official is not None and np.isfinite(fq).all():
            sim = np.flatnonzero((np.abs(df) <= caliper[0]) & (np.abs(dt) <= caliper[1]))
            plans.append(("similar_official", sim[np.argsort(d[sim], kind="stable")][:k], int(sim.size)))
        for kind, sel, n_cand in plans:
            for r, j in enumerate(sel, 1):
                rows.append({"nfl_id": q, "kind": kind, "rank": r, "neighbor_id": int(e.ids[cand[j]]), "position_group": pos, "fold": f,
                             "inductive": bool(ind), "distance": float(d[j]), "distance_pct": float((d < d[j]).mean()),
                             "d_forty": float(df[j]) if np.isfinite(df[j]) else None, "d_ten": float(dt[j]) if np.isfinite(dt[j]) else None,
                             "n_candidates": int(n_cand)})
    return pl.DataFrame(rows, schema=NEIGHBORS.columns) if rows else NEIGHBORS.empty()


def neighbor_overlap(a: pl.DataFrame, b: pl.DataFrame, kind: str = "position") -> dict[int, float]:
    """athlete → Jaccard overlap of his neighbour sets in two neighbour tables (athletes present in both)."""
    def sets(t: pl.DataFrame) -> dict[int, set[int]]:
        g = t.filter(pl.col("kind") == kind).group_by("nfl_id").agg(pl.col("neighbor_id"))
        return {int(i): set(v) for i, v in zip(g["nfl_id"], g["neighbor_id"])}

    sa, sb = sets(a), sets(b)
    return {i: len(sa[i] & sb[i]) / max(len(sa[i] | sb[i]), 1) for i in sa if i in sb}


def distance_lookup(emb: dict[int, FoldEmbeddings]) -> Callable[[int, int], float]:
    """(athlete a, athlete b) → mean of the two inductive distances (each under the encoder that held one of them out)."""
    qf = query_fold(emb)

    def dist(a: int, b: int) -> float:
        out = []
        for x, y in ((a, b), (b, a)):
            if x not in qf:
                continue
            e = emb[qf[x][0]]
            i, j = int(np.searchsorted(e.ids, x)), int(np.searchsorted(e.ids, y))
            if i < e.ids.size and j < e.ids.size and e.ids[i] == x and e.ids[j] == y:
                out.append(float(e.distances(i, np.array([j]))[0]))
        return float(np.mean(out)) if out else NAN

    return dist


# ============================================================================== stability
def stability_table(emb: dict[int, FoldEmbeddings], position: dict[int, str], windows: pl.DataFrame | None = None,
                    overlaps: dict[str, dict[int, float]] | None = None) -> pl.DataFrame:
    """Repeat-attempt stability of every athlete under his query encoder. ``windows``: sequence window flags
    (nfl_id, attempt_order, first_speed, t_first_s) for the recording gaps. ``overlaps``: {'seed': …, 'random': …}
    athlete → neighbour overlap."""
    qf = query_fold(emb)
    gaps: dict[int, tuple[float | None, float | None]] = {}
    if windows is not None and windows.height:
        w = windows.sort("nfl_id", "attempt_order").group_by("nfl_id", maintain_order=True).agg(
            pl.col("first_speed").head(2), pl.col("t_first_s").head(2))
        for i, fs, ts in zip(w["nfl_id"], w["first_speed"], w["t_first_s"]):
            if len(fs) == 2:
                gaps[int(i)] = (None if fs[0] is None or fs[1] is None else abs(fs[1] - fs[0]),
                                None if ts[0] is None or ts[1] is None else abs(ts[1] - ts[0]))
    ov = overlaps or {}
    rows = []
    for q, (f, ind) in sorted(qf.items()):
        e = emb[f]
        i = int(np.searchsorted(e.ids, q))
        pos = position.get(q)
        att = e.attempts_of(i)
        row = {"nfl_id": q, "position_group": pos, "fold": f, "inductive": bool(ind), "n_attempts": int(att.shape[0]), "self_distance": None,
               "self_auc": None, "nearest_other_distance": None, "seed_overlap": ov.get("seed", {}).get(q),
               "random_overlap": ov.get("random", {}).get(q), "first_speed_gap": gaps.get(q, (None, None))[0],
               "start_gap_s": gaps.get(q, (None, None))[1]}
        others = [j for j in range(e.ids.size) if j != i and position.get(int(e.ids[j])) == pos]
        if others:
            firsts = np.stack([e.attempts_of(j)[0] for j in others])
            d_other = np.sqrt(((firsts - att[0]) ** 2).sum(1) / e.E)
            row["nearest_other_distance"] = float(d_other.min())
            if att.shape[0] >= 2:
                d_self = float(np.sqrt(((att[1] - att[0]) ** 2).sum() / e.E))
                row.update(self_distance=d_self, self_auc=float((d_other > d_self).mean() + 0.5 * (d_other == d_self).mean()))
        rows.append(row)
    return pl.DataFrame(rows, schema=STABILITY.columns) if rows else STABILITY.empty()


def seed_overlaps(primary: pl.DataFrame, seeds: pl.DataFrame | None, order: dict[str, int], position: dict[int, str], k: int,
                  queries: Sequence[int] | None = None) -> tuple[dict[int, float], list[str]]:
    """Mean neighbour overlap between the primary TCN-B neighbours and those of every seed-replicate encoder found in
    ``seeds`` (any (scheme, family) whose family names a full TCN embedding). Returns (athlete → overlap, replicates used)."""
    if seeds is None or not seeds.height:
        return {}, []
    used, acc = [], {}
    combos = seeds.select("scheme", "family").unique().sort("scheme", "family")
    for sch, fam in zip(combos["scheme"], combos["family"]):
        if not str(fam).upper().startswith("TCN") or "PC" in str(fam).upper():
            continue
        try:
            nb = neighbor_table(fold_embeddings(seeds, order, fam, sch), position, None, k, queries=queries)
        except ValueError:
            continue
        for i, v in neighbor_overlap(primary, nb).items():
            acc.setdefault(i, []).append(v)
        used.append(f"{sch}/{fam}")
    return {i: float(np.mean(v)) for i, v in acc.items()}, used
