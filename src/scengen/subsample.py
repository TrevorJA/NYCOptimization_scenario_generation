"""Subsampling selectors (methods 4.6, 4.2).

Selects ``n`` realizations from the master ensemble's hazard image ``H``.

  - ``hazard_filling_subsample`` (methods 4.6): the contribution. **LHS +
    nearest-neighbor** space-filling selection over the empirical-CDF-normalized
    hazard image -- a Latin hypercube is drawn over the (uniformized) hazard
    box and each anchor snaps to the nearest not-yet-used scenario. This is the
    ``pick_space_filling_subset`` algorithm named in the methods note (§4.6
    "Implementation"), applied in normalized hazard space so "uniform in hazard
    space" is well-defined under skewed marginals. It is deliberately the
    simplest defensible space-filling design: no annealing, no tuning, and --
    because it does not optimize a discrepancy objective -- L2-star discrepancy
    remains an *independent* build-QC gate (methods 6a) rather than the thing it
    optimized.
  - ``random_subsample`` (methods 4.2): random-without-replacement baseline.

``coverage_metrics``, ``generate_lhs_samples``, and the LHS+NN selection
algorithm are COPIED (not imported) from MOEA-FIND ``src/discovery/analysis.py``
so this repo has no dependency on that repo. Pure numpy/scipy -- no SSI, no
SynHydro, no pywrdrb -- so it is testable on any hazard matrix.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import KDTree, cKDTree
from scipy.stats import rankdata
from scipy.stats.qmc import LatinHypercube, discrepancy


# ---------------------------------------------------------------------------
# Coverage primitives (copied from MOEA-FIND src/discovery/analysis.py)
# ---------------------------------------------------------------------------

def coverage_metrics(points: np.ndarray, lb: np.ndarray, ub: np.ndarray) -> dict:
    """Coverage-quality metrics for a point set (copied from MOEA-FIND).

    Args:
        points: ``(n, d)`` array in original coordinates.
        lb, ub: per-axis bounds used to normalize to the unit cube.

    Returns:
        Dict with ``L2_star_discrepancy`` (lower = more uniform) and
        nearest-neighbor distance statistics (``nn_mean/std/min/max/cv``).
    """
    normed = np.clip((points - lb) / (ub - lb), 0.0, 1.0)
    metrics: dict = {"n_points": len(points), "dimensions": points.shape[1]}
    metrics["L2_star_discrepancy"] = float(discrepancy(normed, method="L2-star"))
    if len(points) > 1:
        tree = KDTree(normed)
        dists, _ = tree.query(normed, k=2)  # k=2: first neighbor is self
        nn = dists[:, 1]
        metrics["nn_mean"] = float(np.mean(nn))
        metrics["nn_std"] = float(np.std(nn))
        metrics["nn_min"] = float(np.min(nn))
        metrics["nn_max"] = float(np.max(nn))
        metrics["nn_cv"] = float(np.std(nn) / np.mean(nn))
    return metrics


def generate_lhs_samples(
    n: int, d: int, lb: np.ndarray, ub: np.ndarray, seed: int = 42
) -> np.ndarray:
    """Latin Hypercube samples in ``[lb, ub]^d`` (copied from MOEA-FIND)."""
    sampler = LatinHypercube(d=d, seed=seed)
    return lb + sampler.random(n=n) * (ub - lb)


# ---------------------------------------------------------------------------
# Hazard-space normalization
# ---------------------------------------------------------------------------

def empirical_cdf_normalize(H: np.ndarray) -> np.ndarray:
    """Map each hazard axis to (0, 1] by its empirical CDF (average ranks).

    Makes "uniform in hazard space" well-defined under skewed marginals: after
    this transform every axis is (approximately) uniform on the unit interval,
    so a Latin hypercube over the unit box fills the hazard manifold evenly.
    Uniform-in-rank selection is quantile stratification: marginally
    *representative* of the pool (the faithful arm).

    Args:
        H: ``(M, d)`` hazard image.

    Returns:
        ``(M, d)`` array with each column ranked into (0, 1].
    """
    H = np.asarray(H, dtype=float)
    M = H.shape[0]
    out = np.empty_like(H)
    for a in range(H.shape[1]):
        out[:, a] = rankdata(H[:, a], method="average") / M
    return out


def minmax_normalize(
    H: np.ndarray, *, lo_pct: float = 0.0, hi_pct: float = 100.0
) -> np.ndarray:
    """Map each hazard axis to [0, 1] by its ABSOLUTE range (optionally robust).

    Unlike :func:`empirical_cdf_normalize`, this preserves the metric's absolute
    spacing, so a Latin hypercube over the unit box targets uniform coverage of
    the *magnitude* range. Uniform-in-magnitude selection over a skewed pool
    over-represents the sparse tails relative to their frequency (the distorted
    arm). ``lo_pct``/``hi_pct`` set robust percentile bounds so a few outliers do
    not dominate the range (``0``/``100`` = full range; ``1``/``99`` = robust).

    Args:
        H: ``(M, d)`` hazard image.
        lo_pct, hi_pct: Per-axis lower/upper percentile bounds for the range.

    Returns:
        ``(M, d)`` array with each column min-max scaled and clipped to [0, 1].
    """
    H = np.asarray(H, dtype=float)
    out = np.empty_like(H)
    for a in range(H.shape[1]):
        lo = np.percentile(H[:, a], lo_pct)
        hi = np.percentile(H[:, a], hi_pct)
        if hi <= lo:
            hi = lo + 1e-12
        out[:, a] = np.clip((H[:, a] - lo) / (hi - lo), 0.0, 1.0)
    return out


# ---------------------------------------------------------------------------
# Selectors
# ---------------------------------------------------------------------------

def random_subsample(H: np.ndarray, n: int, *, seed: int) -> np.ndarray:
    """Random-without-replacement subsample of ``n`` row indices (methods 4.2)."""
    M = len(H)
    if n > M:
        raise ValueError(f"requested {n} but only {M} available")
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(M, size=n, replace=False))


def hazard_filling_subsample(
    H: np.ndarray,
    n: int,
    *,
    seed: int,
    k_pool: int | None = None,
) -> np.ndarray:
    """LHS + nearest-neighbor space-filling subsample over the hazard manifold (methods 4.6).

    Algorithm (``pick_space_filling_subset``, copied from MOEA-FIND
    ``src/discovery/analysis.py`` and applied in normalized hazard space):

        1. Normalize ``H`` per axis to (0, 1] by its empirical CDF, so each axis
           is uniform and a Latin hypercube over the unit box targets uniform
           coverage of the hazard manifold.
        2. Draw ``n`` Latin-hypercube anchors over the unit box.
        3. For each anchor, snap to the nearest scenario not already chosen
           (KDTree query with a small candidate pool; global fallback if the
           pool is exhausted of unused neighbors).

    There is no occupancy/maximin objective and no annealing: the design is the
    deterministic-given-seed nearest-neighbor projection of an LHS plan. Because
    it does not minimize a discrepancy objective, L2-star discrepancy remains an
    independent build-QC gate (methods 6a).

    Args:
        H: ``(M, d)`` hazard image (raw metric values; normalized internally).
        n: Subsample size.
        seed: LHS RNG seed (replicate the design over seeds).
        k_pool: Neighbors queried per anchor before the global fallback; defaults
            to ``min(max(8, n // 4), M)`` (the MOEA-FIND heuristic).

    Returns:
        Sorted integer array of ``n`` selected row indices into ``H``.
    """
    return _lhs_nn_select(empirical_cdf_normalize(H), n, seed=seed, k_pool=k_pool)


def absolute_filling_subsample(
    H: np.ndarray,
    n: int,
    *,
    seed: int,
    lo_pct: float = 0.0,
    hi_pct: float = 100.0,
    k_pool: int | None = None,
) -> np.ndarray:
    """LHS + nearest-neighbor subsample in ABSOLUTE (min-max) hazard space.

    The distorted-arm counterpart to :func:`hazard_filling_subsample`: filling is
    uniform over each axis's *magnitude* range rather than its rank, so the
    selected subset over-represents the sparse tails relative to frequency
    (genuine probability distortion toward extreme-hazard coverage). Over a
    heavy-tailed pool the full-range version can fixate on a few outliers; pass
    ``lo_pct``/``hi_pct`` (e.g. 1/99) for robust bounds.

    Args:
        H: ``(M, d)`` hazard image (raw metric values; normalized internally).
        n: Subsample size.
        seed: LHS RNG seed.
        lo_pct, hi_pct: Robust percentile bounds for the absolute range.
        k_pool: Neighbors queried per anchor before the global fallback.

    Returns:
        Sorted integer array of ``n`` selected row indices into ``H``.
    """
    return _lhs_nn_select(
        minmax_normalize(H, lo_pct=lo_pct, hi_pct=hi_pct), n, seed=seed, k_pool=k_pool
    )


def _lhs_nn_select(
    X: np.ndarray, n: int, *, seed: int, k_pool: int | None = None
) -> np.ndarray:
    """LHS + nearest-neighbor selection over points already normalized to [0, 1]^d.

    Draws ``n`` Latin-hypercube anchors over the unit box and snaps each to the
    nearest not-yet-used point (KDTree query with a small candidate pool; global
    fallback if the pool is exhausted of unused neighbors). Shared by the
    rank-space (faithful) and absolute-space (distorted) selectors so they differ
    only in the normalization of ``X``.
    """
    X = np.asarray(X, dtype=float)
    M, d = X.shape
    if n > M:
        raise ValueError(f"requested {n} but only {M} available")
    if n == M:
        return np.arange(M)

    anchors = generate_lhs_samples(n, d, np.zeros(d), np.ones(d), seed=seed)

    tree = cKDTree(X)
    chosen: list[int] = []
    used: set[int] = set()
    if k_pool is None:
        k_pool = min(max(8, n // 4), M)
    for anchor in anchors:
        _, idxs = tree.query(anchor, k=k_pool)
        idxs = np.atleast_1d(idxs)
        for i in idxs:
            i = int(i)
            if i not in used:
                used.add(i)
                chosen.append(i)
                break
        else:
            # Candidate pool exhausted of unused neighbors; global nearest.
            mask = np.ones(M, dtype=bool)
            mask[list(used)] = False
            remaining = np.where(mask)[0]
            if len(remaining) == 0:
                break
            d2 = ((X[remaining] - anchor) ** 2).sum(axis=1)
            pick = int(remaining[np.argmin(d2)])
            used.add(pick)
            chosen.append(pick)
    return np.sort(np.array(chosen, dtype=int))


def support_point_subsample(H: np.ndarray, n: int, *, seed: int) -> np.ndarray:
    """Energy-distance support points (methods 4.6.1; Mak & Joseph 2018).

    Placeholder for the faithful-x-designed control design (the §4.6.1
    supplement that isolates uniform-coverage benefits from designed-subsampling
    benefits). Not part of the hazard-filling selector.

    Raises:
        NotImplementedError: stub.
    """
    raise NotImplementedError("support_points selector not implemented yet")
