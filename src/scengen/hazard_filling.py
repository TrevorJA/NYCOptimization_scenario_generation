"""Hazard-filling design driver (methods 4.6).

A hazard-filling design owns a **candidate pool** of realizations and selects a
subset of them. It must select rather than generate: hazard coordinates (drought
deficit volume, flood peak magnitude, ...) are *emergent* properties of a
realized flow sequence, so no generator can be asked to produce a realization at
a prescribed drought severity. The Latin-hypercube anchors of the design are
therefore snapped to the nearest real pool member (see :mod:`scengen.subsample`).
Input-space stratification has no such constraint -- forcing parameters are a
knob on the generator -- and so is a generation design, not a subsample.

The selector is deterministic given its seed: LHS + nearest-neighbor snap. No
annealing, no discrepancy objective.

Live pipeline::

    candidate hazard image (streamed to hazard_image.npz at pool generation)
        ->  Olden & Poff redundancy screen  ->  tail-balanced axis set
        ->  LHS + nearest-neighbor selection  ->  selected rows

:func:`select_from_candidate_image` is the entry point. It takes the precomputed
candidate hazard image and never touches the pool timeseries, so it scales to a
very large pool; NYCOptimization materializes the selected realizations on
demand. This module stays pure numpy/scipy apart from the pandas resample in
:func:`daily_to_monthly`.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from . import diagnostics as dg
from . import subsample as ss

#: Operational preference order for picking one representative per redundancy
#: cluster (within-cluster members are interchangeable for coverage; this keeps
#: the operationally-interpretable one). Per tail: magnitude first, then the
#: rate-of-change / duration facets that form the orthogonal cluster(s).
DEFAULT_AXIS_PRIORITY: tuple[str, ...] = (
    "drought_deficit_volume", "drought_onset_rate", "drought_recovery_rate",
    "drought_duration", "drought_peak_depth",
    "flood_peak_magnitude", "flood_pulse_duration", "flood_rise_rate",
)


def _tail_of(axis_name: str) -> str:
    """Map a candidate axis name to its hazard tail (``"dry"`` or ``"wet"``)."""
    return "dry" if axis_name.startswith("drought") else "wet"


def select_balanced_axes(
    representatives: Sequence[str],
    priority: Sequence[str],
    *,
    max_per_tail: int = 2,
) -> list[str]:
    """Tail-balanced final axis set: up to ``max_per_tail`` representatives per tail.

    The Olden & Poff screen determines the redundancy clusters (and one
    representative each); this caps the final set to ``max_per_tail`` axes per
    tail in ``priority`` order, so the design carries a balanced wet+dry set
    (e.g. 2 dry + 2 wet) instead of however many clusters happen to appear. The
    magnitude axis (first in ``priority`` per tail) is always kept; the next
    orthogonal-cluster representative fills the second slot.

    Returns:
        The chosen axis names, dry tail first then wet, each in priority order.
    """
    rank = {a: i for i, a in enumerate(priority)}
    ordered = sorted(representatives, key=lambda a: rank.get(a, len(priority)))
    by_tail: dict[str, list[str]] = {"dry": [], "wet": []}
    for a in ordered:
        t = _tail_of(a)
        if len(by_tail[t]) < max_per_tail:
            by_tail[t].append(a)
    return by_tail["dry"] + by_tail["wet"]


def daily_to_monthly(daily: pd.Series, agg: str = "mean") -> np.ndarray:
    """Aggregate a daily flow Series to monthly (month-start) values."""
    monthly = daily.resample("MS").agg(agg)
    return monthly.to_numpy(dtype=float)


def select_from_candidate_image(
    H_candidates: np.ndarray,
    candidate_axes: Sequence[str],
    n: int,
    *,
    seed: int,
    redundancy_threshold: float = 0.7,
    axis_priority: Sequence[str] = DEFAULT_AXIS_PRIORITY,
    max_per_tail: int = 2,
    selector_space: str = "cdf",
    selector_kwargs: dict | None = None,
) -> dict:
    """Screen a precomputed candidate hazard image, then space-fill-select ``n`` scenarios.

    The single entry point of the hazard-filling design. Given the candidate image ``H_candidates``
    (``M x n_candidates``) already computed for a candidate pool — streamed at pool generation and
    reloaded from ``hazard_image.npz`` — it runs the Olden & Poff redundancy screen (drop degenerate
    axes; keep one operationally-preferred representative per ``|rho_S| >= redundancy_threshold``
    cluster; cap at ``max_per_tail`` per tail) and then the LHS + nearest-neighbor selector on the
    screened sub-image. The axis set is screened per pool rather than hard-coded, so it adapts if the
    generator or pool changes.

    No pool timeseries are read, so this scales to a very large candidate pool; the caller
    materializes only the selected realizations.

    Args:
        H_candidates: ``(M, n_candidates)`` candidate hazard image (raw metric values).
        candidate_axes: Length-``n_candidates`` axis names (columns of ``H_candidates``).
        n: Number of scenarios to select.
        seed: Selector (LHS) RNG seed.
        redundancy_threshold: Spearman ``|rho|`` cut for the redundancy clustering.
        axis_priority: Operational preference order for cluster representatives.
        max_per_tail: Maximum retained axes per hazard tail (dry / wet).
        selector_space: ``"cdf"`` (faithful, rank space) or ``"abs"`` (distorted, magnitude space).
        selector_kwargs: Extra kwargs forwarded to the selector (e.g. ``k_pool``).

    Returns:
        Dict with ``selected_rows`` (sorted indices), ``chosen_axes`` (the screened axis set used for
        selection), ``candidate_axes``, ``H_candidates``, ``screen`` (spread + clusters), and
        ``coverage`` (selected vs random L2-star on the chosen sub-image).
    """
    H_full = np.asarray(H_candidates, dtype=float)
    candidate_axes = list(candidate_axes)

    # Olden & Poff screen: drop degenerate axes, keep one representative per cluster.
    spread = dg.per_metric_spread(H_full, candidate_axes)
    kept = [a for a in candidate_axes if not spread[a]["degenerate"]]
    keep_idx = [candidate_axes.index(a) for a in kept]
    clusters = dg.spearman_clusters(
        H_full[:, keep_idx], kept, threshold=redundancy_threshold, priority=axis_priority
    )
    # Tail-balanced final set: up to max_per_tail representatives per tail.
    chosen_axes = select_balanced_axes(
        clusters["representatives"], axis_priority, max_per_tail=max_per_tail
    )
    chosen_idx = [candidate_axes.index(a) for a in chosen_axes]
    H_sel = H_full[:, chosen_idx]

    # Selection space: "cdf" (faithful, rank space) or "abs" (distorted, absolute magnitude space).
    # Coverage QC is always reported in CDF/rank space so the two arms share a discrepancy scale.
    if selector_space == "cdf":
        sel = ss.hazard_filling_subsample(H_sel, n, seed=seed, **(selector_kwargs or {}))
    elif selector_space == "abs":
        sel = ss.absolute_filling_subsample(H_sel, n, seed=seed, **(selector_kwargs or {}))
    else:
        raise ValueError(f"unknown selector_space={selector_space!r}")
    X = ss.empirical_cdf_normalize(H_sel)
    lb, ub = np.zeros(X.shape[1]), np.ones(X.shape[1])
    rand = ss.random_subsample(H_sel, n, seed=seed)
    coverage = {
        "selected_L2_star": ss.coverage_metrics(X[sel], lb, ub)["L2_star_discrepancy"],
        "random_L2_star": ss.coverage_metrics(X[rand], lb, ub)["L2_star_discrepancy"],
    }
    return {
        "selected_rows": sel,
        "chosen_axes": chosen_axes,
        "candidate_axes": candidate_axes,
        "H_candidates": H_full,
        "screen": {"spread": spread, "clusters": clusters["clusters"], "representatives": chosen_axes},
        "coverage": coverage,
    }
