"""Hazard-space subsampling selectors (methods 4.6, 4.2).

Selects ``n`` realizations from a candidate pool's hazard image ``H``.

**Why these selectors SELECT rather than GENERATE.** Hazard coordinates (drought
deficit volume, flood peak magnitude, ...) are *emergent* properties of a
realized flow sequence: no generator can be asked to emit a realization at a
prescribed drought severity, because severity is only known after the sequence
exists. A hazard-space design therefore has nothing to generate *to* -- it must
SELECT FROM a finite candidate pool, and its Latin-hypercube anchors must snap
to the nearest real pool member. This nearest-neighbor snap is the whole reason
the selector exists.

The contrast is with *input*-space (forcing-parameter) stratification: the
forcing parameters ``theta`` ARE a knob on the generator, so an input-space
design draws an LHS over ``theta`` and generates one realization per design
point. It never subsamples, and there is nothing to snap to. That design lives
in the generator (``forcing_space.sample_harmonic_forcing(method="lhs")``), not
in this module.

Selectors here:

  - ``hazard_filling_subsample`` (methods 4.6): the contribution. LHS +
    nearest-neighbor space-filling selection over the empirical-CDF-normalized
    hazard image, so "uniform in hazard space" is well-defined under skewed
    marginals (the faithful / rank-space arm).
  - ``absolute_filling_subsample``: the same selector in absolute (min-max)
    magnitude space -- a retained non-campaign sensitivity, not a campaign arm.
  - ``random_subsample`` (methods 4.2): the random-without-replacement baseline
    that the coverage-vs-random QC gate compares against.

The selector is deterministic given its seed: an LHS plan projected onto the
pool by nearest neighbor. There is no discrepancy objective, no annealing, and
no tuning -- so L2-star discrepancy stays an *independent* build-QC gate
(methods 6a) rather than the quantity the selector optimized.

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
        3. Snap each anchor to the nearest scenario not already chosen (KDTree
           query over the candidate pool; global fallback if the local pool is
           exhausted of unused neighbors). The snap is forced by the nature of
           hazard coordinates: they are emergent from a realized sequence, so
           the design can only select an existing pool member near the anchor,
           never generate a realization at the anchor.

    The result is the deterministic-given-seed nearest-neighbor projection of an
    LHS plan; no discrepancy objective is optimized, so L2-star discrepancy
    remains an independent build-QC gate (methods 6a).

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

    The distorted counterpart to :func:`hazard_filling_subsample`, retained as a
    non-campaign sensitivity: filling is uniform over each axis's *magnitude*
    range rather than its rank, so the selected subset over-represents the sparse
    tails relative to their frequency (genuine probability distortion toward
    extreme-hazard coverage). Over a heavy-tailed pool the full-range version can
    fixate on a few outliers; pass ``lo_pct``/``hi_pct`` (e.g. 1/99) for robust
    bounds. Same LHS + nearest-neighbor snap, same emergent-coordinate rationale
    -- only the normalization of the hazard image differs.

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
    """LHS + nearest-neighbor selection over pool points already normalized to [0, 1]^d.

    Draws ``n`` Latin-hypercube anchors over the unit box and snaps each to the
    nearest not-yet-used pool point (KDTree query over a local candidate pool;
    global fallback if that pool is exhausted of unused neighbors). The snap is
    what makes this a *selection* rather than a *generation* design -- see the
    module docstring on why hazard coordinates cannot be generated to.

    Shared by the rank-space (faithful) and absolute-space (distorted) hazard
    selectors, which differ only in the normalization applied to ``X``. It is not
    used for input-space stratification: forcing parameters are a generator knob,
    so that design generates one realization per LHS point instead.

    Args:
        X: ``(M, d)`` pool coordinates already normalized to the unit box.
        n: Number of points to select.
        seed: LHS RNG seed.
        k_pool: Neighbors queried per anchor before the global fallback; defaults
            to ``min(max(8, n // 4), M)`` (the MOEA-FIND heuristic).

    Returns:
        Sorted integer array of ``n`` selected row indices into ``X``.
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
