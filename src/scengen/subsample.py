"""Hazard-space subsampling selectors.

Selects ``n`` realizations from a candidate pool's hazard image ``H``. The
specification is ``scenario_design_methods.md`` (§4.3 selector, §6 diagnostics).

**Why these selectors SELECT rather than GENERATE.** Hazard coordinates (drought
magnitude, flood peak discharge, ...) are *emergent* properties of a
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

  - ``absolute_filling_subsample``: the CAMPAIGN selector of the hazard-filling
    design. LHS + nearest-neighbor space-filling selection over the ABSOLUTE
    (min-max range-scaled) hazard image, so filling is uniform over each axis's
    *magnitude* range. Because the pool's hazard marginals are strongly
    right-skewed, this draws selected members from the sparse severe corners far
    more often than their pool frequency: severe drought and flood conditions are
    over-represented relative to their probability under the generator. That is
    the deliberate distribution shift the study tests, not a defect.
  - ``cdf_filling_subsample``: the same selector over the
    empirical-CDF-normalized (rank) hazard image. Filling uniformly in rank
    reproduces the pool's marginal frequencies and distorts only the joint
    dependence among axes. Retained as a NON-CAMPAIGN sensitivity that isolates
    how much of any hazard-filling effect is attributable to absolute-space tail
    over-representation specifically.
  - ``random_subsample``: the random-without-replacement baseline that the
    coverage QC compares against.

The selector is deterministic given its seed: an LHS plan projected onto the
pool by nearest neighbor. There is no discrepancy objective, no annealing, and
no tuning -- so L2-star discrepancy stays an *independent* build-QC diagnostic
rather than the quantity the selector optimized.

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

    After this transform every axis is (approximately) uniform on the unit
    interval, so a Latin hypercube over the unit box fills the *rank* image of
    the hazard manifold evenly. Uniform-in-rank selection is quantile
    stratification: it reproduces the pool's marginal frequencies and distorts
    only the joint dependence among axes. This is the geometry of the
    non-campaign sensitivity, not of the campaign selector.

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


#: Campaign robust percentile bounds for the absolute-space normalization.
#: The bounds must be CENTRAL order statistics, not sample extremes: the sample
#: min/max of a right-skewed hazard metric are non-convergent extreme order
#: statistics, so a full-range box would (i) depend on the pool size P (a bigger
#: pool widens the range and strengthens the tail distortion, entangling the
#: intervention's strength with a nuisance sizing parameter), (ii) differ
#: materially across the K pool re-rolls (breaking draw commensurability), and
#: (iii) let a single outlier compress the bulk of the pool into a corner of the
#: box. The p1/p99 quantiles are root-P-consistent, so the box converges to a
#: fixed population functional. On the zero-inflated dry event axes p1 collapses
#: to the natural zero automatically. Members outside the bounds clip to the box
#: faces (a bounded, <= 1% + 1% atom per axis, reported as build-QC) and remain
#: selectable there.
ROBUST_LO_PCT: float = 1.0
ROBUST_HI_PCT: float = 99.0


def robust_range_bounds(
    H: np.ndarray, *, lo_pct: float = ROBUST_LO_PCT, hi_pct: float = ROBUST_HI_PCT
) -> tuple[np.ndarray, np.ndarray]:
    """Per-axis robust range bounds ``(lo, hi)`` for the absolute normalization.

    Single source of the bounds used by :func:`minmax_normalize`, the coverage
    QC, and the normalization build-QC, so all three always agree.

    Args:
        H: ``(M, d)`` hazard image.
        lo_pct, hi_pct: Percentile bounds (campaign default p1/p99; see
            :data:`ROBUST_LO_PCT`).

    Returns:
        Two length-``d`` arrays ``(lo, hi)`` with ``hi > lo`` guaranteed.
    """
    H = np.asarray(H, dtype=float)
    lo = np.percentile(H, lo_pct, axis=0)
    hi = np.percentile(H, hi_pct, axis=0)
    hi = np.where(hi <= lo, lo + 1e-12, hi)
    return lo, hi


def minmax_normalize(
    H: np.ndarray, *, lo_pct: float = ROBUST_LO_PCT, hi_pct: float = ROBUST_HI_PCT
) -> np.ndarray:
    """Map each hazard axis to [0, 1] by its robust ABSOLUTE range.

    Unlike :func:`empirical_cdf_normalize`, this preserves the metric's absolute
    spacing, so a Latin hypercube over the unit box targets uniform coverage of
    the *magnitude* range and no single axis dominates the distance while spacing
    within an axis stays proportional to physical magnitude. Uniform-in-magnitude
    selection over a skewed pool over-represents the sparse tails relative to
    their frequency -- the deliberate distribution shift the study tests. This is
    the geometry of the CAMPAIGN selector.

    Bounds default to the robust campaign percentiles (p1/p99; rationale at
    :data:`ROBUST_LO_PCT`). Pass ``lo_pct=0, hi_pct=100`` for the full-range
    variant, retained only as a sensitivity.

    Args:
        H: ``(M, d)`` hazard image.
        lo_pct, hi_pct: Per-axis percentile bounds for the range.

    Returns:
        ``(M, d)`` array with each column scaled and clipped to [0, 1].
    """
    H = np.asarray(H, dtype=float)
    lo, hi = robust_range_bounds(H, lo_pct=lo_pct, hi_pct=hi_pct)
    return np.clip((H - lo) / (hi - lo), 0.0, 1.0)


# ---------------------------------------------------------------------------
# Selectors
# ---------------------------------------------------------------------------

def random_subsample(H: np.ndarray, n: int, *, seed: int) -> np.ndarray:
    """Random-without-replacement subsample of ``n`` row indices."""
    M = len(H)
    if n > M:
        raise ValueError(f"requested {n} but only {M} available")
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(M, size=n, replace=False))


def absolute_filling_subsample(
    H: np.ndarray,
    n: int,
    *,
    seed: int,
    lo_pct: float = ROBUST_LO_PCT,
    hi_pct: float = ROBUST_HI_PCT,
    k_pool: int | None = None,
) -> np.ndarray:
    """LHS + nearest-neighbor subsample in ABSOLUTE (robust min-max) hazard space.

    The CAMPAIGN selector of the hazard-filling design. Algorithm
    (``pick_space_filling_subset``, copied from MOEA-FIND
    ``src/discovery/analysis.py`` and applied in normalized hazard space):

        1. Scale ``H`` per axis to [0, 1] by its robust pool range (p1/p99
           campaign default; rationale at :data:`ROBUST_LO_PCT`), so distances
           are in absolute, range-scaled magnitude units and the box is a stable
           functional of the population rather than of the pool's sample
           extremes.
        2. Draw ``n`` Latin-hypercube anchors over the unit box.
        3. Snap each anchor to the nearest scenario not already chosen (KDTree
           query over the candidate pool; global fallback if the local pool is
           exhausted of unused neighbors). The snap is forced by the nature of
           hazard coordinates: they are emergent from a realized sequence, so
           the design can only select an existing pool member near the anchor,
           never generate a realization at the anchor.

    Filling the *range* uniformly over a right-skewed pool over-represents the
    sparse severe corners relative to their pool frequency -- the deliberate
    distribution shift the study tests. The full-range variant
    (``lo_pct=0, hi_pct=100``) is retained only as a sensitivity: it lets a
    single outlier compress the bulk of the pool and ties the box to the pool
    size P.

    The result is the deterministic-given-seed nearest-neighbor projection of an
    LHS plan; no discrepancy objective is optimized, so L2-star discrepancy
    remains an independent build-QC diagnostic.

    Args:
        H: ``(M, d)`` hazard image (raw metric values; normalized internally).
        n: Subsample size.
        seed: LHS RNG seed (replicate the design over seeds).
        lo_pct, hi_pct: Percentile bounds for the absolute range (campaign
            default p1/p99).
        k_pool: Neighbors queried per anchor before the global fallback; defaults
            to ``min(max(8, n // 4), M)`` (the MOEA-FIND heuristic).

    Returns:
        Sorted integer array of ``n`` selected row indices into ``H``.
    """
    return _lhs_nn_select(
        minmax_normalize(H, lo_pct=lo_pct, hi_pct=hi_pct), n, seed=seed, k_pool=k_pool
    )


def cdf_filling_subsample(
    H: np.ndarray,
    n: int,
    *,
    seed: int,
    k_pool: int | None = None,
) -> np.ndarray:
    """LHS + nearest-neighbor subsample in empirical-CDF (rank) hazard space.

    The retained NON-CAMPAIGN sensitivity, not the campaign selector. It differs
    from :func:`absolute_filling_subsample` only in the normalization of the
    hazard image: each axis is mapped to (0, 1] by its empirical CDF, so filling
    is uniform in *rank* rather than in magnitude. That reproduces the pool's
    marginal frequencies and distorts only the joint dependence among axes, which
    is exactly what makes it the sensitivity that isolates how much of any
    hazard-filling effect comes from absolute-space tail over-representation.
    Same LHS + nearest-neighbor snap, same emergent-coordinate rationale.

    Args:
        H: ``(M, d)`` hazard image (raw metric values; normalized internally).
        n: Subsample size.
        seed: LHS RNG seed.
        k_pool: Neighbors queried per anchor before the global fallback.

    Returns:
        Sorted integer array of ``n`` selected row indices into ``H``.
    """
    return _lhs_nn_select(empirical_cdf_normalize(H), n, seed=seed, k_pool=k_pool)


def _lhs_nn_select(
    X: np.ndarray, n: int, *, seed: int, k_pool: int | None = None
) -> np.ndarray:
    """LHS + nearest-neighbor selection over pool points already normalized to [0, 1]^d.

    Draws ``n`` Latin-hypercube anchors over the unit box and snaps each to the
    nearest not-yet-used pool point (KDTree query over a local candidate pool;
    global fallback if that pool is exhausted of unused neighbors). The snap is
    what makes this a *selection* rather than a *generation* design -- see the
    module docstring on why hazard coordinates cannot be generated to.

    Shared by the absolute-space (campaign) and rank-space (non-campaign
    sensitivity) hazard selectors, which differ only in the normalization applied
    to ``X``. It is not used for input-space stratification: forcing parameters
    are a generator knob, so that design generates one realization per LHS point
    instead.

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
