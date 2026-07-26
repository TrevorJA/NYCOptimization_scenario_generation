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
                     wired status-quo selector, ``subsample._lhs_nn_select``).
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

Pure numpy/scipy/pandas — no SSI, no SynHydro, no pywrdrb — so the comparison
runs on any staged hazard image, at laptop or HPC scale.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.sparse.csgraph import minimum_spanning_tree
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
        info: Selector-specific diagnostics — ``snap_distances`` (anchor-to-
            selected distance per anchor, LHS selectors), ``grid_resolution``
            and ``n_occupied_cells`` (``eps_cell``).
    """

    rows: np.ndarray
    info: dict = field(default_factory=dict)


def select_random(X: np.ndarray, n: int, *, seed: int) -> SelectorResult:
    """Random without replacement — the null rule every alternative must beat."""
    return SelectorResult(rows=ss.random_subsample(X, n, seed=seed))


def select_lhs_nn(X: np.ndarray, n: int, *, seed: int) -> SelectorResult:
    """LHS anchors + greedy nearest-unused-neighbor snap (the wired selector)."""
    rows = ss._lhs_nn_select(X, n, seed=seed)
    anchors = ss.generate_lhs_samples(
        n, X.shape[1], np.zeros(X.shape[1]), np.ones(X.shape[1]), seed=seed
    )
    # Greedy pairing is order-dependent, so re-derive each anchor's realized
    # partner by re-running the pairing order: nearest selected member works as
    # the reporting proxy (exact for the non-contended anchors).
    d = cdist(anchors, X[rows])
    return SelectorResult(rows=rows, info={"snap_distances": d.min(axis=1)})


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
