"""Ensemble-quality / build-QC diagnostics (methods 6a).

These are *build-QC* diagnostics: they verify how well a subsampled ensemble
covers the hazard manifold it was drawn from. They are NOT evidence of an
outcome advantage (the overfitting-gap / stability hypotheses of methods 6b need
MOEA re-evaluation and live in ``../NYCOptimization``).

Because the wired hazard-filling selector (LHS + nearest-neighbor) does NOT
optimize a discrepancy objective, L2-star discrepancy is here an *independent*
coverage gate -- reported against the distribution of random subsets of the same
size so a coverage advantage is shown relative to chance, not asserted.

This module is pure numpy/scipy and operates on a hazard image ``H`` (raw metric
values) plus the indices of a selected subset; it has no SSI / SynHydro / pywrdrb
dependency. Plotting lives in the diagnostics scripts, not here.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr

from .subsample import coverage_metrics, empirical_cdf_normalize, random_subsample


# ---------------------------------------------------------------------------
# Hazard-image persistence (handoff from the NYCOpt subsample step)
# ---------------------------------------------------------------------------

def save_hazard_image(
    path: str | Path,
    *,
    H: np.ndarray,
    hazard_axes,
    realization_ids,
    selected_rows,
    reference_start: str,
    chosen_axes=None,
) -> Path:
    """Persist a pool hazard image + the selected rows for offline diagnostics.

    The NYCOptimization subsample step (which holds the historical reference for
    the SSI fit) computes ``H`` and writes it here so the scengen diagnostics
    are reproducible and decoupled from pywrdrb.

    Args:
        path: Output ``.npz`` path.
        H: ``(M, m)`` pool hazard image of the full candidate axes (raw values).
        hazard_axes: Length-``m`` candidate-axis names (columns of ``H``).
        realization_ids: Length-``M`` pool realization ids aligned with ``H`` rows.
        selected_rows: Indices into ``H`` of the selected subset.
        reference_start: Start date the SSI reference fit was stamped with
            (date-convention provenance; :func:`load_hazard_image` rejects
            images that lack it, so pre-convention artifacts self-label as
            stale rather than silently mixing conventions).
        chosen_axes: The screened subset of ``hazard_axes`` actually used for
            selection (defaults to all of ``hazard_axes``).

    Returns:
        The written path.
    """
    path = Path(path)
    if chosen_axes is None:
        chosen_axes = list(hazard_axes)
    np.savez(
        path,
        H=np.asarray(H, dtype=float),
        hazard_axes=np.asarray(list(hazard_axes), dtype=object),
        chosen_axes=np.asarray(list(chosen_axes), dtype=object),
        realization_ids=np.asarray(list(realization_ids), dtype=int),
        selected_rows=np.asarray(list(selected_rows), dtype=int),
        reference_start=np.asarray(str(reference_start), dtype=object),
    )
    return path


def load_hazard_image(path: str | Path) -> dict:
    """Load a hazard image written by :func:`save_hazard_image`.

    Raises:
        ValueError: If the file lacks the ``reference_start`` provenance field —
            it predates the truthful January date convention and its hazard
            coordinates must not be mixed with current-convention artifacts.
    """
    path = Path(path)
    data = np.load(path, allow_pickle=True)
    if "reference_start" not in data:
        raise ValueError(
            f"{path} lacks 'reference_start' provenance: it was written before the "
            f"truthful January date convention and is stale. Regenerate the hazard image."
        )
    axes = [str(a) for a in data["hazard_axes"]]
    chosen = [str(a) for a in data["chosen_axes"]] if "chosen_axes" in data else list(axes)
    return {
        "H": data["H"],
        "hazard_axes": axes,
        "chosen_axes": chosen,
        "realization_ids": data["realization_ids"].astype(int),
        "selected_rows": data["selected_rows"].astype(int),
        "reference_start": str(data["reference_start"]),
    }


# ---------------------------------------------------------------------------
# Coverage diagnostics
# ---------------------------------------------------------------------------

def mst_mean_edge(X: np.ndarray) -> float:
    """Mean edge length of the Euclidean minimum spanning tree of ``X``.

    A larger mean MST edge at fixed ``n`` indicates points are more spread out
    (less clumped) over the manifold.
    """
    if len(X) < 2:
        return 0.0
    tree = minimum_spanning_tree(squareform(pdist(X)))
    edges = tree.toarray()
    nz = edges[edges > 0]
    return float(nz.mean()) if nz.size else 0.0


def hazard_effective_sample_size(
    X: np.ndarray, *, ideal_spacing: float | None = None, bandwidth_factor: float = 0.33
) -> dict:
    """Effective sample size of a point set as a redundancy measure (lower = more clumped).

    Measures how many *effectively distinct* points a design occupies via Gaussian-kernel overlap.
    For points ``X`` (already normalized to the unit box ``[0, 1]^d``) and a kernel
    ``K_ij = exp(-||x_i - x_j||^2 / (2 h^2))``::

        ESS = N^2 / sum_ij K_ij

    A perfectly separated design has ``K -> I`` and ``ESS -> N``; a fully clumped design has
    ``K -> 1`` everywhere and ``ESS -> 1``. The bandwidth defaults to ``bandwidth_factor * N^{-1/d}``
    -- a fraction of the ideal uniform inter-point spacing, set *below* that spacing so a uniform
    design scores ``ESS ~ N`` and redundancy shows as ``ESS < N``; the same ``h`` is used for all
    designs at a given ``(N, d)`` so they are directly comparable. (Note: in higher ``d`` the
    kernel-overlap ESS is a *weak* discriminator -- non-uniformity shows up far more sharply in
    L2-star discrepancy; ESS is best read as a corroborating redundancy summary.) Pass
    ``ideal_spacing`` to fix ``h`` directly across designs of differing ``N``.

    The kernel-matrix participation ratio ``(sum lambda)^2 / sum lambda^2`` is reported as a
    secondary, spectral view of the same redundancy.

    Args:
        X: ``(N, d)`` points normalized to the unit box.
        ideal_spacing: Override bandwidth ``h`` directly; default ``bandwidth_factor * N^{-1/d}``.
        bandwidth_factor: Fraction of the ideal spacing used for ``h`` when ``ideal_spacing`` is None.

    Returns:
        Dict with ``ess`` (kernel-overlap ESS), ``ess_fraction`` (``ess / N``), ``participation_ratio``,
        ``bandwidth``, ``n``, ``dimensions``.
    """
    X = np.asarray(X, dtype=float)
    n, d = X.shape
    if n < 2:
        return {
            "ess": float(n), "ess_fraction": 1.0, "participation_ratio": float(n),
            "bandwidth": float("nan"), "n": int(n), "dimensions": int(d),
        }
    h = float(ideal_spacing) if ideal_spacing is not None else bandwidth_factor * n ** (-1.0 / d)
    sq = squareform(pdist(X, metric="sqeuclidean"))
    K = np.exp(-sq / (2.0 * h**2))
    ess = float(n**2 / K.sum())
    lam = np.linalg.eigvalsh(K)
    lam = lam[lam > 0]
    pr = float(lam.sum() ** 2 / np.square(lam).sum()) if lam.size else float(n)
    return {
        "ess": ess,
        "ess_fraction": ess / n,
        "participation_ratio": pr,
        "bandwidth": h,
        "n": int(n),
        "dimensions": int(d),
    }


def expected_random_discrepancy(
    n: int,
    m: int,
    *,
    n_boot: int = 500,
    seed: int = 0,
    pool_X: np.ndarray | None = None,
) -> dict:
    """Bootstrap the L2-star discrepancy of a random design at given ``(n, m)``.

    Args:
        n: Subset size.
        m: Hazard dimensionality.
        n_boot: Bootstrap replicates.
        seed: RNG seed.
        pool_X: If given, draw random *subsets of this normalized pool* (the
            decision-relevant baseline: random subsampling, i.e. the
            fixed-probabilistic alternative). If ``None``, draw uniform random
            points in the unit cube (a pure-geometry reference).

    Returns:
        Dict with ``mean``, ``std``, and the raw ``samples`` array.
    """
    rng = np.random.default_rng(seed)
    lb, ub = np.zeros(m), np.ones(m)
    samples = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        if pool_X is not None:
            idx = rng.choice(len(pool_X), size=n, replace=False)
            pts = pool_X[idx]
        else:
            pts = rng.random((n, m))
        samples[b] = coverage_metrics(pts, lb, ub)["L2_star_discrepancy"]
    return {"mean": float(samples.mean()), "std": float(samples.std()), "samples": samples}


@dataclass(frozen=True)
class CoverageReport:
    """Coverage diagnostics for a selected subset's hazard coordinates.

    Attributes:
        n: Subset size.
        m: Hazard dimensionality.
        l2_star: Selected subset's L2-star discrepancy (lower = more uniform).
        nn_min: Minimum nearest-neighbor distance (separation/maximin).
        nn_mean: Mean nearest-neighbor distance.
        nn_cv: Coefficient of variation of NN distances (lower = more even).
        mst_mean: Mean Euclidean MST edge length.
        random_l2_mean: Mean L2-star of random subsets of the same size.
        random_l2_std: SD of that random distribution.
        discrepancy_percentile: Fraction of random subsets with L2-star <= the
            selected subset's. Near 0 = selected is more uniform than almost all
            random draws (good); near 1 = worse than random.
    """

    n: int
    m: int
    l2_star: float
    nn_min: float
    nn_mean: float
    nn_cv: float
    mst_mean: float
    random_l2_mean: float
    random_l2_std: float
    discrepancy_percentile: float


def coverage_report(
    H: np.ndarray,
    selected_rows,
    *,
    n_boot: int = 500,
    seed: int = 0,
) -> CoverageReport:
    """Coverage QC for a subset, in empirical-CDF-normalized hazard space.

    Args:
        H: ``(M, m)`` pool hazard image (raw metric values; normalized here).
        selected_rows: Indices into ``H`` of the selected subset.
        n_boot: Bootstrap replicates for the random-design baseline.
        seed: RNG seed for the bootstrap.

    Returns:
        A :class:`CoverageReport`.
    """
    X = empirical_cdf_normalize(H)
    m = X.shape[1]
    sel = np.asarray(selected_rows, dtype=int)
    Xsub = X[sel]
    lb, ub = np.zeros(m), np.ones(m)

    cm = coverage_metrics(Xsub, lb, ub)
    rnd = expected_random_discrepancy(len(sel), m, n_boot=n_boot, seed=seed, pool_X=X)
    pct = float((rnd["samples"] <= cm["L2_star_discrepancy"]).mean())

    return CoverageReport(
        n=len(sel),
        m=m,
        l2_star=cm["L2_star_discrepancy"],
        nn_min=cm.get("nn_min", 0.0),
        nn_mean=cm.get("nn_mean", 0.0),
        nn_cv=cm.get("nn_cv", 0.0),
        mst_mean=mst_mean_edge(Xsub),
        random_l2_mean=rnd["mean"],
        random_l2_std=rnd["std"],
        discrepancy_percentile=pct,
    )


def marginal_gaps(H: np.ndarray, selected_rows) -> np.ndarray:
    """Largest per-axis gap in the selected subset's normalized marginal.

    For each hazard axis, the selected points are mapped to (0, 1] by the pool's
    empirical CDF and sorted; the statistic is the largest gap between
    consecutive points including the [0, 1] boundaries. Smaller = no large
    uncovered stretch on that axis (good marginal coverage). A uniform spread of
    ``n`` points has gaps near ``1/(n+1)``.

    Returns:
        Length-``m`` array of per-axis maximum gaps.
    """
    X = empirical_cdf_normalize(H)
    sel = np.asarray(selected_rows, dtype=int)
    out = np.empty(X.shape[1])
    for a in range(X.shape[1]):
        pts = np.sort(X[sel, a])
        edges = np.concatenate([[0.0], pts, [1.0]])
        out[a] = float(np.max(np.diff(edges)))
    return out


def redundancy_screen(
    H: np.ndarray, hazard_axes, *, threshold: float = 0.7
) -> dict:
    """Spearman redundancy screen on the hazard axes (methods 3.3, exploratory).

    Flags axis pairs with ``|rho_S| >= threshold`` (Olden & Poff 2003 redundancy
    cut). Reported on the master pool's hazard image so the axis set can be
    pruned before scaling up.

    Returns:
        Dict with the ``axes`` list, the Spearman correlation matrix ``rho``, and
        the list of ``redundant_pairs`` ``(axis_i, axis_j, rho)`` above threshold.
    """
    H = np.asarray(H, dtype=float)
    axes = list(hazard_axes)
    rho, _ = spearmanr(H)
    rho = np.atleast_2d(rho)
    pairs = []
    for i in range(len(axes)):
        for j in range(i + 1, len(axes)):
            if abs(rho[i, j]) >= threshold:
                pairs.append((axes[i], axes[j], float(rho[i, j])))
    return {"axes": axes, "rho": rho, "redundant_pairs": pairs}


# ---------------------------------------------------------------------------
# Hazard-axis screening (Olden & Poff 2003 — candidate-pool reduction)
# ---------------------------------------------------------------------------

def per_metric_spread(H: np.ndarray, hazard_axes) -> dict:
    """Per-axis spread / degeneracy screen.

    Flags axes with **near-zero spread** — a near-constant axis, a zero-IQR
    axis, or one whose mass piles almost entirely at a single value. Skew and
    the modal-mass fraction are reported as descriptive statistics but are NOT
    drop criteria: hazard metrics are strongly right-skewed by nature, and the
    axis policy keeps every non-degenerate descriptor.

    Returns:
        Dict mapping each axis to its ``std``, ``iqr``, ``skew``, ``zero_frac``
        (fraction equal to the modal value), and a ``degenerate`` flag
        (near-zero spread only).
    """
    from scipy.stats import skew as _skew

    H = np.asarray(H, dtype=float)
    out = {}
    for a, name in enumerate(hazard_axes):
        col = H[:, a]
        q75, q25 = np.percentile(col, [75, 25])
        iqr = float(q75 - q25)
        std = float(col.std())
        sk = float(_skew(col)) if std > 1e-12 else 0.0
        vals, counts = np.unique(np.round(col, 9), return_counts=True)
        zero_frac = float(counts.max() / len(col))
        out[name] = {
            "std": std,
            "iqr": iqr,
            "skew": sk,
            "zero_frac": zero_frac,
            "degenerate": bool(std < 1e-9 or iqr == 0.0 or zero_frac > 0.95),
        }
    return out


def spearman_clusters(
    H: np.ndarray, hazard_axes, *, threshold: float = 0.7, priority=None
) -> dict:
    """Average-linkage Spearman clustering of axes (Olden & Poff 2003 framing).

    A **diagnostic only**: the correlation structure (matrix and ``1 - |rho_S|``
    cluster tree) is reported alongside the axis selection, never used to reduce
    the selection axis set (that is
    :func:`scengen.hazard_filling.screen_hazard_axes`, which prunes only
    near-duplicates). Clusters axes on distance ``1 - |rho_S|`` cut so any pair
    with ``|rho_S| >= threshold`` lands in one cluster (cut height
    ``1 - threshold``).

    The representative is chosen by ``priority`` when given (a list of axis names
    in preference order — the operationally-preferred member of each cluster is
    kept, since within-cluster members are statistically interchangeable for
    coverage). Falls back to the highest-spread (widest-IQR) member.

    Returns:
        Dict with ``clusters`` (list of axis-name lists), ``representatives``
        (one axis per cluster), and the ``rho`` matrix.
    """
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform

    H = np.asarray(H, dtype=float)
    axes = list(hazard_axes)
    rho, _ = spearmanr(H)
    rho = np.atleast_2d(rho)
    d = 1.0 - np.abs(rho)
    np.fill_diagonal(d, 0.0)
    d = np.clip((d + d.T) / 2.0, 0.0, None)

    if len(axes) == 1:
        labels = np.array([1])
    else:
        Z = linkage(squareform(d, checks=False), method="average")
        labels = fcluster(Z, t=1.0 - threshold, criterion="distance")

    spread = per_metric_spread(H, axes)
    prio = list(priority) if priority else []

    def _rep(members):
        ranked = [m for m in prio if m in members]
        if ranked:
            return ranked[0]
        return max(members, key=lambda m: spread[m]["iqr"])

    clusters, reps = [], []
    for c in sorted(set(labels)):
        members = [axes[i] for i in range(len(axes)) if labels[i] == c]
        clusters.append(members)
        reps.append(_rep(members))
    return {"clusters": clusters, "representatives": reps, "rho": rho}


def pca_effective_dimension(H: np.ndarray) -> dict:
    """PCA effective dimension of the hazard image.

    Reports the **participation ratio** (effective rank) of the correlation-matrix
    eigen-spectrum, ``PR = (sum lambda)^2 / sum(lambda^2)``, as the primary
    effective dimensionality. PR equals ``p`` for ``p`` uncorrelated axes and
    collapses toward 1 as axes become collinear, so it is the robust measure of
    "how many dimensions must the subsample fill" (drives the ``N >= q^m``
    budget). The broken-stick retained count is reported too, but it pathologically
    under-counts flat (already-de-correlated) spectra and should not be used on a
    post-screen axis set.

    Returns:
        Dict with ``explained_variance_ratio``, ``participation_ratio`` (float),
        ``broken_stick`` expectations, ``broken_stick_dimension``, and
        ``effective_dimension`` (``round(participation_ratio)``).
    """
    H = np.asarray(H, dtype=float)
    p = H.shape[1]
    Z = (H - H.mean(axis=0)) / np.where(H.std(axis=0) > 1e-12, H.std(axis=0), 1.0)
    evals = np.clip(np.sort(np.linalg.eigvalsh(np.corrcoef(Z, rowvar=False)))[::-1], 0.0, None)
    evr = evals / evals.sum()
    pr = float((evals.sum() ** 2) / np.sum(evals ** 2))

    bs = np.array([(1.0 / p) * sum(1.0 / k for k in range(i + 1, p + 1)) for i in range(p)])
    bs_dim = 0
    for i in range(p):
        if evr[i] > bs[i]:
            bs_dim += 1
        else:
            break
    return {
        "explained_variance_ratio": evr,
        "participation_ratio": pr,
        "broken_stick": bs,
        "broken_stick_dimension": max(1, int(bs_dim)),
        "effective_dimension": int(round(pr)),
    }
