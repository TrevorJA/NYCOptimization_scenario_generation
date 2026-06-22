"""Subsampling selectors (methods 4.6, 4.2).

Selects ``n`` realizations from the master ensemble's hazard image ``H``.

  - ``hazard_filling_subsample`` (methods 4.6): the contribution. A *stratified
    maximin space-filling design with cLHS-style marginal (quantile-stratum)
    conditioning* -- NOT cLHS (the parent-correlation term that defines cLHS is
    omitted by construction, since it is antithetical to uniform hazard
    coverage). Marginal quantile-stratification after Minasny & McBratney (2006);
    the maximin / phi_p separation term after Morris & Mitchell (1995) and
    Johnson et al. (1990).
  - ``random_subsample`` (methods 4.2): random-without-replacement baseline.

``coverage_metrics`` and ``generate_lhs_samples`` are COPIED (not imported) from
MOEA-FIND ``src/discovery/analysis.py`` so this repo has no dependency on that
repo; the simulated-annealing selector is net-new. Pure numpy/scipy -- no SSI,
no SynHydro, no pywrdrb -- so it is testable on any hazard matrix.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import KDTree
from scipy.spatial.distance import pdist
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
    so equal-width strata are equal-probability strata.

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


def _occupancy_cost(Xsub: np.ndarray, n_strata: int) -> float:
    """Sum over axes of |stratum occupancy - target| (cLHS marginal condition).

    ``Xsub`` is already empirical-CDF-normalized, so equal-width bins are
    equal-probability strata; the per-stratum target is ``n / n_strata``.
    """
    n, d = Xsub.shape
    target = n / n_strata
    cost = 0.0
    for a in range(d):
        bins = np.clip((Xsub[:, a] * n_strata).astype(int), 0, n_strata - 1)
        counts = np.bincount(bins, minlength=n_strata)
        cost += np.abs(counts - target).sum()
    return float(cost)


def _maximin_phi_p(Xsub: np.ndarray, p: int) -> float:
    """phi_p maximin surrogate (Morris & Mitchell 1995); lower = better spread."""
    if len(Xsub) < 2:
        return 0.0
    d = pdist(Xsub)
    d = np.maximum(d, 1e-12)
    return float((np.sum(d ** (-p))) ** (1.0 / p))


def hazard_filling_subsample(
    H: np.ndarray,
    n: int,
    *,
    seed: int,
    n_strata: int | None = None,
    maximin_weight: float = 0.3,
    p: int = 15,
    iters: int = 20000,
    t0: float = 1.0,
    cooling: float = 0.9995,
) -> np.ndarray:
    r"""Stratified maximin space-filling subsample over the hazard manifold (methods 4.6).

    Minimizes, by simulated annealing over size-``n`` subsets ``T`` of the
    empirical-CDF-normalized hazard image::

        Phi(T) = occupancy_mismatch(T) + maximin_weight * phi_p(T)

    The occupancy term is the cLHS marginal (quantile-stratum) condition; the
    phi_p term is the maximin separation. There is NO parent-correlation term
    (the deliberate distortion toward uniform hazard coverage).

    Args:
        H: ``(M, d)`` hazard image (raw metric values; normalized internally).
        n: Subsample size.
        seed: RNG seed (the selector is stochastic; replicate over seeds).
        n_strata: Quantile strata per axis (default ``n`` -> one point per stratum).
        maximin_weight: Weight on the phi_p separation term relative to occupancy.
        p: phi_p exponent (larger -> closer to true maximin).
        iters: Simulated-annealing iteration budget.
        t0: Initial temperature.
        cooling: Geometric cooling factor applied each iteration.

    Returns:
        Sorted integer array of ``n`` selected row indices into ``H``.
    """
    X = empirical_cdf_normalize(H)
    M = X.shape[0]
    if n > M:
        raise ValueError(f"requested {n} but only {M} available")
    if n == M:
        return np.arange(M)
    n_strata = n_strata or n
    rng = np.random.default_rng(seed)

    def cost(idx: np.ndarray) -> float:
        sub = X[idx]
        return _occupancy_cost(sub, n_strata) + maximin_weight * _maximin_phi_p(sub, p)

    sel = rng.choice(M, size=n, replace=False)
    in_set = np.zeros(M, dtype=bool)
    in_set[sel] = True
    cur = cost(sel)
    best_sel, best_cost = sel.copy(), cur
    temp = t0
    outside = np.where(~in_set)[0]

    for _ in range(iters):
        # Propose: swap one selected member for one non-member.
        rem_pos = int(rng.integers(n))
        add_pos = int(rng.integers(len(outside)))
        old_idx, new_idx = sel[rem_pos], outside[add_pos]
        sel[rem_pos] = new_idx
        cand = cost(sel)
        delta = cand - cur
        if delta <= 0 or rng.random() < np.exp(-delta / max(temp, 1e-12)):
            cur = cand
            outside[add_pos] = old_idx  # keep the swap; old member now outside
            if cur < best_cost:
                best_cost, best_sel = cur, sel.copy()
        else:
            sel[rem_pos] = old_idx       # revert
        temp *= cooling

    return np.sort(best_sel)


def support_point_subsample(H: np.ndarray, n: int, *, seed: int) -> np.ndarray:
    """Energy-distance support points (methods 4.6.1; Mak & Joseph 2018).

    Placeholder for the faithful-x-designed control design; not part of the
    initial hazard_filling draft.

    Raises:
        NotImplementedError: stub.
    """
    raise NotImplementedError("support_points selector not implemented yet")
