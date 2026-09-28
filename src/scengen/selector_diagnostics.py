"""Selector-comparison diagnostics for the hazard-filling design.

Candidate SELECTION RULES for choosing ``n`` pool members whose hazard
coordinates cover the (absolute, robust range-scaled) hazard manifold, plus the
metric battery that compares them. The campaign selector is chosen FROM this
comparison; the module exists so that choice is made from measured diagnostics
on a real candidate pool rather than asserted.

All selectors operate on a pool image already normalized to the unit box (the
caller picks the geometry — for the campaign question that is the robust
absolute normalization, ``subsample.minmax_normalize``) and return a
:class:`SelectorResult` with the selected row indices plus selector-specific
diagnostics (snap distances, cell resolution, ...).

Selectors:

  - ``random``     — random without replacement (the null / control rule).
  - ``lhs_nn``     — LHS anchors + greedy nearest-unused-neighbor snap (the
                     wired status-quo selector, ``subsample.lhs_nn_assignment``).
  - ``lhs_assign`` — the same LHS anchors, but assigned to pool members by an
                     optimal one-to-one matching (min total squared distance,
                     Hungarian algorithm). Isolates the greedy snap's
                     order-dependence: same anchors, globally optimal pairing.
  - ``maximin``    — greedy maximin distance selection (Kennard–Stone-type:
                     start at the pool point farthest from the pool mean, then
                     repeatedly add the point maximizing the minimum distance to
                     the already-selected set). Deterministic; the DOE-standard
                     comparator, known to load the hull/boundary.
  - ``eps_cell``   — epsilon-cell selection: grid the unit box at the coarsest
                     per-axis resolution with at least ``n`` OCCUPIED cells,
                     draw ``n`` occupied cells uniformly at random, and take one
                     representative per cell (the member nearest the cell
                     center). Uniform over the empirical manifold's occupied
                     support at resolution epsilon — no anchor can land off the
                     manifold, and no two selected members share a cell (an
                     epsilon-net-like separation guarantee). The cell-grid
                     device is the coverage analogue of epsilon-dominance
                     archiving (Laumanns et al. 2002).

``knn_min_sum_assignment`` solves the exact minimum-total-displacement
assignment of the same anchors to pool rows on a k-nearest-candidate graph, with
a certificate of global optimality, so the greedy snap's gap to the exact
solution is measured rather than assumed (the design formalization in
NYCOptimization ``docs/notes/methods/hf_design_metrics.md``).

Pure numpy/scipy/pandas — no SSI, no SynHydro, no pywrdrb — so the comparison
runs on any staged hazard image, at laptop or HPC scale.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment, linprog
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import min_weight_full_bipartite_matching, minimum_spanning_tree
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
from scipy.stats import ks_2samp

from . import subsample as ss


###############################################################################
# Selector results + rules
###############################################################################

@dataclass(frozen=True)
class SelectorResult:
    """Outcome of one selection: row indices plus selector-specific diagnostics.

    Attributes:
        rows: Sorted selected row indices into the pool image.
        info: Selector-specific diagnostics — ``snap_distances`` (distance from
            each anchor to the member assigned to it, in anchor emission order,
            LHS selectors), ``n_fallback`` (``lhs_nn``), ``grid_resolution``
            and ``n_occupied_cells`` (``eps_cell``).
    """

    rows: np.ndarray
    info: dict = field(default_factory=dict)


def select_random(X: np.ndarray, n: int, *, seed: int) -> SelectorResult:
    """Random without replacement — the null rule every alternative must beat."""
    return SelectorResult(rows=ss.random_subsample(X, n, seed=seed))


def select_lhs_nn(X: np.ndarray, n: int, *, seed: int) -> SelectorResult:
    """LHS anchors + greedy nearest-unused-neighbor snap (the wired selector)."""
    a = ss.lhs_nn_assignment(X, n, seed=seed)
    return SelectorResult(
        rows=np.sort(a.rows),
        info={"snap_distances": a.displacement, "n_fallback": a.n_fallback},
    )


def select_lhs_assign(X: np.ndarray, n: int, *, seed: int) -> SelectorResult:
    """LHS anchors + optimal one-to-one assignment (min total squared distance).

    Same anchor plan as ``lhs_nn``; the pairing is the Hungarian solution over
    the full anchor-by-pool cost matrix instead of a greedy sequential snap, so
    the result is independent of anchor order. Deterministic given the seed.
    """
    d = X.shape[1]
    anchors = ss.generate_lhs_samples(n, d, np.zeros(d), np.ones(d), seed=seed)
    cost = cdist(anchors, X, metric="sqeuclidean")
    rr, cc = linear_sum_assignment(cost)
    snap = np.sqrt(cost[rr, cc])
    return SelectorResult(rows=np.sort(cc), info={"snap_distances": snap})


@dataclass(frozen=True)
class SparseAssignment:
    """Exact minimum-total-displacement assignment of targets to pool rows.

    Attributes:
        rows: ``(n,)`` pool row assigned to each target, in target order.
        displacement: ``(n,)`` Euclidean target-to-member distance.
        total: Sum of ``displacement``.
        k: Neighbours per target of the graph on which ``rows`` was solved.
        certified: True when ``total`` is proven to be the global optimum over
            every injective assignment into the whole pool.
        slack: ``max_i (u_i - r_i(k))`` for the optimal dual potentials ``u``
            of the sparse problem; non-positive when certified.
        ladder: One record per rung tried: ``k``, ``feasible``, ``total``,
            ``lp_total``, ``slack``, ``certified``.
    """

    rows: np.ndarray
    displacement: np.ndarray
    total: float
    k: int
    certified: bool
    slack: float
    ladder: tuple[dict, ...]


def knn_min_sum_assignment(
    targets: np.ndarray,
    X: np.ndarray,
    *,
    k_ladder: Sequence[int],
    tree: cKDTree | None = None,
) -> SparseAssignment:
    """Exact min-sum assignment of targets to pool rows on a k-nearest graph.

    Solves ``min_sigma sum_i ||u_i - x_sigma(i)||`` over injective ``sigma`` by
    min-weight full bipartite matching on the graph joining each target to its
    ``k`` nearest pool rows, growing ``k`` along ``k_ladder`` until the solution
    is certified globally optimal. Certificate (linear-programming duality):
    the assignment LP on the sparse graph is solved for its dual potentials,
    ``u_i`` per target and ``v_j <= 0`` per candidate in the graph (``v_j = 0``
    for candidates outside it). Every excluded edge ``(i, j)`` has cost at
    least the k-th neighbour radius ``r_i(k)``, so ``u_i <= r_i(k)`` for all
    ``i`` makes the sparse dual feasible for the complete problem and, by
    strong duality, proves the sparse optimum is the global optimum. A rung
    with no full matching is skipped; the ladder must reach ``n`` so a full
    matching exists at its last rung.

    Weights are offset by 1.0 because the matcher treats zero weights as absent
    edges; totals are recomputed from the true distances.

    Args:
        targets: ``(n, d)`` target points in the unit box.
        X: ``(M, d)`` pool coordinates in the unit box.
        k_ladder: Increasing neighbour counts to try (capped at ``M``).
        tree: Prebuilt ``cKDTree(X)`` to reuse; built here if None.

    Returns:
        The assignment at the first certified rung (or the last rung).
    """
    targets = np.asarray(targets, dtype=float)
    X = np.asarray(X, dtype=float)
    n = targets.shape[0]
    M = X.shape[0]
    if n > M:
        raise ValueError(f"requested {n} but only {M} available")
    ladder_k = sorted({int(min(k, M)) for k in k_ladder if k >= 1})
    if not ladder_k or ladder_k[-1] < n:
        raise ValueError(f"k_ladder must reach n={n} (got {list(k_ladder)})")
    if tree is None:
        tree = cKDTree(X)

    records: list[dict] = []
    best: SparseAssignment | None = None
    for k in ladder_k:
        dist, idx = tree.query(targets, k=k)
        dist = np.asarray(dist, dtype=float).reshape(n, k)
        idx = np.asarray(idx, dtype=int).reshape(n, k)
        uniq, col = np.unique(idx.ravel(), return_inverse=True)
        infeasible = {"k": k, "feasible": False, "total": float("nan"), "certified": False}
        if len(uniq) < n:  # fewer distinct candidates than targets: no full matching
            records.append(infeasible)
            continue
        graph = csr_matrix(
            (dist.ravel() + 1.0, (np.repeat(np.arange(n), k), col.ravel())),
            shape=(n, len(uniq)),
        )
        try:
            r, c = min_weight_full_bipartite_matching(graph)
        except ValueError:
            records.append(infeasible)
            continue
        if len(r) != n:  # the matcher saturated the smaller side, not the targets
            records.append(infeasible)
            continue
        order = np.argsort(r)
        rows = uniq[c[order]]
        displacement = np.sqrt(((targets - X[rows]) ** 2).sum(axis=1))
        total = float(displacement.sum())
        lp_total, slack = _dual_certificate(dist, col, n_cols=len(uniq))
        certified = bool(
            k >= M
            or (slack <= 1e-9 and abs(lp_total - total) <= 1e-7 * max(1.0, total))
        )
        records.append({"k": k, "feasible": True, "total": total, "lp_total": lp_total,
                        "slack": slack, "certified": certified})
        best = SparseAssignment(
            rows=rows, displacement=displacement, total=total, k=k,
            certified=certified, slack=float(slack), ladder=(),
        )
        if certified:
            break
    if best is None:  # unreachable when the ladder reaches n (Hall's condition)
        raise RuntimeError("no rung of the k ladder admitted a full matching")
    return SparseAssignment(
        rows=best.rows, displacement=best.displacement, total=best.total, k=best.k,
        certified=best.certified, slack=best.slack, ladder=tuple(records),
    )


def _dual_certificate(dist: np.ndarray, col: np.ndarray, *, n_cols: int) -> tuple[float, float]:
    """LP objective and certificate slack ``max_i (u_i - r_i(k))`` of a kNN graph.

    Args:
        dist: ``(n, k)`` distances from each target to its k nearest rows.
        col: ``(n, k)`` compact column index of those rows.
        n_cols: Number of distinct rows in the graph.

    Returns:
        ``(lp_total, slack)``; ``slack <= 0`` certifies global optimality.
    """
    n, k = dist.shape
    ne = n * k
    edge = np.arange(ne)
    a_eq = csr_matrix((np.ones(ne), (np.repeat(np.arange(n), k), edge)), shape=(n, ne))
    a_ub = csr_matrix((np.ones(ne), (col.ravel(), edge)), shape=(n_cols, ne))
    res = linprog(dist.ravel(), A_ub=a_ub, b_ub=np.ones(n_cols), A_eq=a_eq,
                  b_eq=np.ones(n), bounds=(0, None), method="highs")
    if not res.success:
        return float("nan"), float("inf")
    u = np.asarray(res.eqlin.marginals, dtype=float)
    return float(res.fun), float(np.max(u - dist[:, -1]))


def select_maximin(X: np.ndarray, n: int, *, seed: int) -> SelectorResult:
    """Greedy maximin (Kennard–Stone-type) selection over the pool points.

    Anchor-free: selects actual pool members directly, so nothing can land off
    the manifold. Deterministic — ``seed`` is accepted for API uniformity and
    ignored; replicate variance comes from the pool re-roll alone.
    """
    X = np.asarray(X, dtype=float)
    first = int(np.argmax(((X - X.mean(axis=0)) ** 2).sum(axis=1)))
    chosen = [first]
    mind = ((X - X[first]) ** 2).sum(axis=1)
    for _ in range(n - 1):
        nxt = int(np.argmax(mind))
        chosen.append(nxt)
        mind = np.minimum(mind, ((X - X[nxt]) ** 2).sum(axis=1))
    return SelectorResult(rows=np.sort(np.array(chosen, dtype=int)))


def select_eps_cell(X: np.ndarray, n: int, *, seed: int) -> SelectorResult:
    """Epsilon-cell selection: uniform over the occupied cells of a hazard grid.

    Calibrates the per-axis grid resolution ``g`` to the smallest integer with
    at least ``n`` occupied cells (epsilon = 1/g), draws ``n`` occupied cells
    uniformly without replacement, and represents each drawn cell by the member
    nearest its center (ties broken by row index). Uniform over the manifold's
    occupied support at resolution epsilon; guarantees one member per cell.
    """
    X = np.asarray(X, dtype=float)
    M, d = X.shape
    if n > M:
        raise ValueError(f"requested {n} but only {M} available")

    def occupied(g: int) -> dict[tuple, np.ndarray]:
        cells = np.minimum((X * g).astype(int), g - 1)
        keys = [tuple(c) for c in cells]
        groups: dict[tuple, list[int]] = {}
        for i, k in enumerate(keys):
            groups.setdefault(k, []).append(i)
        return {k: np.array(v) for k, v in groups.items()}

    # Smallest g with >= n occupied cells (occupancy is monotone in g).
    lo_g, hi_g = 1, 2
    while len(occupied(hi_g)) < n:
        hi_g *= 2
        if hi_g > 4096:
            raise ValueError("cannot reach n occupied cells; pool too small/degenerate")
    while lo_g < hi_g:
        mid = (lo_g + hi_g) // 2
        if len(occupied(mid)) >= n:
            hi_g = mid
        else:
            lo_g = mid + 1
    g = hi_g
    groups = occupied(g)

    rng = np.random.default_rng(seed)
    keys = sorted(groups)  # deterministic ordering before the seeded draw
    drawn = rng.choice(len(keys), size=n, replace=False)
    chosen = []
    for j in drawn:
        key = keys[j]
        members = groups[key]
        center = (np.array(key) + 0.5) / g
        dist2 = ((X[members] - center) ** 2).sum(axis=1)
        chosen.append(int(members[np.argmin(dist2)]))
    return SelectorResult(
        rows=np.sort(np.array(chosen, dtype=int)),
        info={"grid_resolution": int(g), "n_occupied_cells": int(len(keys))},
    )


#: Name -> rule. ``fn(X_normalized, n, seed=...) -> SelectorResult``.
SELECTORS: dict[str, Callable[..., SelectorResult]] = {
    "random": select_random,
    "lhs_nn": select_lhs_nn,
    "lhs_assign": select_lhs_assign,
    "maximin": select_maximin,
    "eps_cell": select_eps_cell,
}


###############################################################################
# Metric battery
###############################################################################

def _mst_edge_stats(P: np.ndarray) -> dict[str, float]:
    """Minimum-spanning-tree edge statistics of a point set (uniformity facet)."""
    D = cdist(P, P)
    mst = minimum_spanning_tree(D).toarray()
    edges = mst[mst > 0]
    return {
        "mst_edge_mean": float(edges.mean()),
        "mst_edge_min": float(edges.min()),
        "mst_edge_cv": float(edges.std() / edges.mean()),
    }


def selection_metrics(
    H: np.ndarray,
    rows: np.ndarray,
    axes: Sequence[str],
    *,
    lo_pct: float = ss.ROBUST_LO_PCT,
    hi_pct: float = ss.ROBUST_HI_PCT,
) -> dict[str, float]:
    """The scalar metric battery for one selected set on one pool image.

    Covers the four facets of the selector comparison: coverage uniformity
    (L2-star in the abs and rank geometries; MST edge stats and minimum
    separation in abs geometry), tail enrichment (mean per-axis share above the
    pool P90 and the any-axis P90 corner share — the deliberate distribution
    shift, read against the ~0.10 of an unbiased rule), marginal distortion
    (mean KS distance to the pool), and the dry zero-event atom (share of
    selected members with no drought event vs the pool's share).

    Args:
        H: ``(M, d)`` pool sub-image on the chosen axes (raw metric values).
        rows: Selected row indices.
        axes: Axis names (used to identify the dry axes for the atom metric).
        lo_pct, hi_pct: Robust bounds of the abs geometry.

    Returns:
        Flat ``{metric: value}`` dict (one row of the comparison table).
    """
    H = np.asarray(H, dtype=float)
    rows = np.asarray(rows, dtype=int)
    Xabs = ss.minmax_normalize(H, lo_pct=lo_pct, hi_pct=hi_pct)
    Xcdf = ss.empirical_cdf_normalize(H)
    lb, ub = np.zeros(H.shape[1]), np.ones(H.shape[1])

    cov_abs = ss.coverage_metrics(Xabs[rows], lb, ub)
    cov_cdf = ss.coverage_metrics(Xcdf[rows], lb, ub)

    p90 = np.percentile(H, 90, axis=0)
    tail_share = float(np.mean([np.mean(H[rows, k] > p90[k]) for k in range(H.shape[1])]))
    corner_share = float(np.mean((H[rows] > p90).any(axis=1)))
    ks_mean = float(np.mean(
        [ks_2samp(H[rows, k], H[:, k]).statistic for k in range(H.shape[1])]
    ))

    dry = [k for k, a in enumerate(axes) if str(a).startswith("drought")]
    out = {
        "L2_star_abs": float(cov_abs["L2_star_discrepancy"]),
        "L2_star_cdf": float(cov_cdf["L2_star_discrepancy"]),
        "nn_min_abs": float(cov_abs.get("nn_min", 0.0)),
        "nn_cv_abs": float(cov_abs.get("nn_cv", 0.0)),
        "tail_share_p90": tail_share,
        "corner_share_p90": corner_share,
        "ks_mean_vs_pool": ks_mean,
        **_mst_edge_stats(Xabs[rows]),
    }
    if dry:
        zero_sel = float(np.mean((H[np.ix_(rows, dry)] == 0.0).all(axis=1)))
        zero_pool = float(np.mean((H[:, dry] == 0.0).all(axis=1)))
        out["zero_event_share_selected"] = zero_sel
        out["zero_event_share_pool"] = zero_pool
    return out


def per_axis_selection_metrics(
    H: np.ndarray,
    rows: np.ndarray,
    axes: Sequence[str],
    *,
    lo_pct: float = ss.ROBUST_LO_PCT,
    hi_pct: float = ss.ROBUST_HI_PCT,
) -> dict[str, dict[str, float]]:
    """Per-axis marginal coverage + tail enrichment of one selected set.

    The mechanism metric of the design: LHS anchors stratify EVERY axis into N
    bins regardless of dimension, so the coverage guarantee is per-axis marginal
    (not joint). This measures, per axis in the campaign scaled coordinates:
    the KS distance of the selected marginal to uniform on [0, 1], the 1-D
    L2-star discrepancy, the largest marginal gap, and the tail share above the
    pool P90 (unbiased rule ≈ 0.10).

    Args:
        H: ``(M, d)`` pool sub-image on the chosen axes (raw metric values).
        rows: Selected row indices.
        axes: Axis names (columns of ``H``).
        lo_pct, hi_pct: Robust bounds of the abs geometry.

    Returns:
        ``{axis: {ks_to_uniform, star_1d, max_gap, tail_share_p90}}``.
    """
    from scipy.stats.qmc import discrepancy

    H = np.asarray(H, dtype=float)
    rows = np.asarray(rows, dtype=int)
    X = ss.minmax_normalize(H, lo_pct=lo_pct, hi_pct=hi_pct)
    p90 = np.percentile(H, 90, axis=0)

    out: dict[str, dict[str, float]] = {}
    for k, name in enumerate(axes):
        x = np.sort(X[rows, k])
        n = len(x)
        up = np.arange(1, n + 1) / n
        lo = np.arange(0, n) / n
        ks = float(max(np.max(up - x), np.max(x - lo)))
        star = float(discrepancy(x.reshape(-1, 1), method="L2-star"))
        edges = np.concatenate([[0.0], x, [1.0]])
        out[str(name)] = {
            "ks_to_uniform": ks,
            "star_1d": star,
            "max_gap": float(np.max(np.diff(edges))),
            "tail_share_p90": float(np.mean(H[rows, k] > p90[k])),
        }
    return out


def distance_concentration(
    X: np.ndarray,
    snap_distances: np.ndarray,
    *,
    seed: int = 0,
    n_pairs: int = 20000,
) -> dict[str, float]:
    """Snap-distance concentration relative to random pool-pair distances.

    In high dimension all pairwise distances concentrate, so a raw snap-distance
    mean is not comparable across dimensions. The ratio (mean snap distance /
    mean random-pair distance in the same normalized space) is: values well
    below 1 mean anchors land materially closer to their snapped member than a
    random pool point would be, i.e. the snap is still informative at this
    dimension.

    Args:
        X: ``(M, d)`` pool coordinates normalized to the unit box.
        snap_distances: Anchor-to-selected distances from an LHS selector.
        seed: RNG seed for the random pool pairs.
        n_pairs: Random pairs sampled to estimate the mean pair distance.

    Returns:
        Dict with ``mean_snap``, ``mean_random_pair``, ``concentration_ratio``.
    """
    X = np.asarray(X, dtype=float)
    rng = np.random.default_rng(seed)
    i = rng.integers(0, len(X), size=n_pairs)
    j = rng.integers(0, len(X), size=n_pairs)
    keep = i != j
    pair_d = np.sqrt(((X[i[keep]] - X[j[keep]]) ** 2).sum(axis=1))
    mean_snap = float(np.mean(snap_distances))
    mean_pair = float(np.mean(pair_d))
    return {
        "mean_snap": mean_snap,
        "mean_random_pair": mean_pair,
        "concentration_ratio": mean_snap / mean_pair,
    }


def snap_axis_contributions(
    X: np.ndarray, n: int, axes: Sequence[str], *, seed: int
) -> dict[str, float]:
    """Per-axis share of the squared anchor-to-member snap displacement.

    The weighting diagnostic: correlated axes implicitly weight their shared
    hazard concept in the Euclidean snap distance. This measures each axis's
    mean fractional contribution to the squared displacement between each anchor
    and the member the greedy rule assigned to it. Shares sum to 1 across axes.

    Args:
        X: ``(M, d)`` pool coordinates normalized to the unit box.
        n: Number of anchors (the ``lhs_nn`` ensemble size).
        axes: Axis names (columns of ``X``).
        seed: The LHS seed of the selection.

    Returns:
        ``{axis: mean fractional contribution}``.
    """
    X = np.asarray(X, dtype=float)
    d = X.shape[1]
    a = ss.lhs_nn_assignment(X, n, seed=seed)
    diff2 = (a.targets - X[a.rows]) ** 2
    tot = diff2.sum(axis=1, keepdims=True)
    shares = np.divide(diff2, tot, out=np.full_like(diff2, 1.0 / d), where=tot > 0)
    mean_share = shares.mean(axis=0)
    return {str(a): float(s) for a, s in zip(axes, mean_share)}


def jaccard(a: np.ndarray, b: np.ndarray) -> float:
    """Jaccard similarity of two selected-index sets."""
    sa, sb = set(np.asarray(a).tolist()), set(np.asarray(b).tolist())
    return float(len(sa & sb) / len(sa | sb)) if sa | sb else float("nan")


def pairwise_jaccard(row_sets: Sequence[np.ndarray]) -> float:
    """Mean pairwise Jaccard similarity of selected-index sets (seed stability)."""
    sets = [set(np.asarray(r).tolist()) for r in row_sets]
    if len(sets) < 2:
        return float("nan")
    vals = [
        len(a & b) / len(a | b)
        for i, a in enumerate(sets) for b in sets[i + 1:]
    ]
    return float(np.mean(vals))


###############################################################################
# Comparison driver
###############################################################################

def run_selector_comparison(
    H: np.ndarray,
    axes: Sequence[str],
    n: int,
    *,
    seeds: Sequence[int],
    selectors: Sequence[str] = tuple(SELECTORS),
    lo_pct: float = ss.ROBUST_LO_PCT,
    hi_pct: float = ss.ROBUST_HI_PCT,
    pool_label: str = "pool",
) -> tuple[pd.DataFrame, dict]:
    """Run every selector at every seed on one pool image; return the tidy table.

    The comparison is run in the campaign geometry: the pool sub-image is
    normalized once with the robust absolute bounds and every selector selects
    in that space, so differences are attributable to the selection rule alone.

    Args:
        H: ``(M, d)`` pool sub-image on the chosen axes (raw metric values).
        axes: Axis names for ``H``'s columns.
        n: Ensemble size to select.
        seeds: Selector seeds (deterministic selectors repeat identically; their
            across-seed spread is a structural zero, which is itself reported).
        selectors: Names from :data:`SELECTORS`.
        lo_pct, hi_pct: Robust bounds of the selection geometry.
        pool_label: Tag written into the ``pool`` column (e.g. a sub-pool id).

    Returns:
        ``(table, details)``: ``table`` has one row per (selector, seed) with the
        :func:`selection_metrics` battery; ``details[selector]`` holds per-seed
        ``rows`` and selector ``info`` (snap distances, cell resolution) plus
        ``jaccard_across_seeds``.
    """
    H = np.asarray(H, dtype=float)
    X = ss.minmax_normalize(H, lo_pct=lo_pct, hi_pct=hi_pct)

    records: list[dict] = []
    details: dict[str, dict] = {}
    for name in selectors:
        fn = SELECTORS[name]
        per_seed = []
        for seed in seeds:
            res = fn(X, n, seed=int(seed))
            m = selection_metrics(H, res.rows, axes, lo_pct=lo_pct, hi_pct=hi_pct)
            records.append({
                "pool": pool_label, "selector": name, "seed": int(seed),
                "n": int(n), "lo_pct": lo_pct, "hi_pct": hi_pct, **m,
            })
            per_seed.append(res)
        details[name] = {
            "rows": [r.rows for r in per_seed],
            "info": [r.info for r in per_seed],
            "jaccard_across_seeds": pairwise_jaccard([r.rows for r in per_seed]),
        }
    return pd.DataFrame.from_records(records), details
