"""Hazard-filling design driver (``scenario_design_methods.md`` §4.3).

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


#: Geometries the coverage QC is reported in. The campaign selector fills the
#: ``"abs"`` geometry and the non-campaign sensitivity fills ``"cdf"``, so both
#: are always reported: each selector wins in its own geometry, and the
#: cross-geometry entry is what shows the trade.
_COVERAGE_GEOMETRIES: tuple[str, ...] = ("cdf", "abs")

#: Random size-n subsets drawn to form the coverage null distribution.
_COVERAGE_NULL_DRAWS: int = 100


def coverage_qc(
    H_sel: np.ndarray,
    selected_rows: np.ndarray,
    *,
    seed: int,
    n_null: int = _COVERAGE_NULL_DRAWS,
    lo_pct: float = ss.ROBUST_LO_PCT,
    hi_pct: float = ss.ROBUST_HI_PCT,
) -> dict:
    """L2-star coverage of the selected set against a random-subset null, per geometry.

    The selector optimizes no discrepancy objective, so discrepancy is an
    independent build-QC diagnostic of what the selection achieved. A single
    random comparator is not enough to read it: L2-star at size ``n`` has its own
    sampling spread, so the selected value is placed inside the null distribution
    of ``n_null`` random size-``n`` subsets rather than against one draw.

    Both geometries are reported. The full pool sub-image is normalized once per
    geometry and the selected / null rows are taken from it, so every set is
    scored on the same unit box (bounds ``[0, 1]^d``) and the numbers are
    comparable across sets and across designs.

    Args:
        H_sel: ``(M, d)`` pool sub-image on the chosen axes (raw metric values).
        selected_rows: Row indices of the selected set into ``H_sel``.
        seed: Base seed; null draw ``i`` uses ``seed * 1000 + i``, so the null is
            deterministic and reproducible from the design's selector seed.
        n_null: Number of random size-``n`` subsets forming the null.
        lo_pct, hi_pct: Percentile bounds of the ``"abs"`` geometry — must match
            the bounds the selector filled (campaign default p1/p99).

    Returns:
        JSON-serializable dict with ``n_selected``, ``n_pool``, ``n_null`` and a
        ``geometries`` map ``{"cdf"|"abs": {...}}`` holding ``selected_L2_star``,
        ``null_mean``, ``null_std``, ``null_min``, ``null_max`` and
        ``percentile`` (percent of null draws with a LOWER discrepancy than the
        selected set; lower percentile = more uniform than the null).
    """
    H_sel = np.asarray(H_sel, dtype=float)
    sel = np.asarray(selected_rows, dtype=int)
    n = int(len(sel))
    normalizers = {
        "cdf": ss.empirical_cdf_normalize,
        "abs": lambda H: ss.minmax_normalize(H, lo_pct=lo_pct, hi_pct=hi_pct),
    }

    geometries: dict[str, dict] = {}
    for geom in _COVERAGE_GEOMETRIES:
        X = normalizers[geom](H_sel)
        lb, ub = np.zeros(X.shape[1]), np.ones(X.shape[1])

        def l2(rows: np.ndarray) -> float:
            return float(ss.coverage_metrics(X[rows], lb, ub)["L2_star_discrepancy"])

        selected_l2 = l2(sel)
        null = np.array(
            [l2(ss.random_subsample(X, n, seed=seed * 1000 + i)) for i in range(n_null)]
        )
        geometries[geom] = {
            "selected_L2_star": selected_l2,
            "null_mean": float(null.mean()),
            "null_std": float(null.std()),
            "null_min": float(null.min()),
            "null_max": float(null.max()),
            "percentile": float(np.mean(null < selected_l2) * 100.0),
        }
    return {
        "n_selected": n,
        "n_pool": int(H_sel.shape[0]),
        "n_null": int(n_null),
        "geometries": geometries,
    }


def screen_axes(
    H_candidates: np.ndarray,
    candidate_axes: Sequence[str],
    *,
    redundancy_threshold: float = 0.7,
    axis_priority: Sequence[str] = DEFAULT_AXIS_PRIORITY,
    max_per_tail: int = 2,
) -> dict:
    """Olden & Poff redundancy screen: candidate axes -> tail-balanced final set.

    Drops degenerate axes (near-zero spread), clusters the survivors on
    ``1 - |rho_S|`` cutting at ``|rho_S| >= redundancy_threshold``, keeps one
    operationally-preferred representative per cluster, and caps the final set at
    ``max_per_tail`` axes per hazard tail. Shared by the live selection driver
    (:func:`select_from_candidate_image`) and the selector diagnostics, so both
    always screen identically.

    Args:
        H_candidates: ``(M, n_candidates)`` candidate hazard image.
        candidate_axes: Length-``n_candidates`` axis names.
        redundancy_threshold: Spearman ``|rho|`` cut for the clustering.
        axis_priority: Operational preference order for cluster representatives.
        max_per_tail: Maximum retained axes per hazard tail (dry / wet).

    Returns:
        Dict with ``representatives`` (the tail-balanced final axis list),
        ``spread`` (per-axis degeneracy stats) and ``clusters``.
    """
    H_full = np.asarray(H_candidates, dtype=float)
    candidate_axes = list(candidate_axes)
    spread = dg.per_metric_spread(H_full, candidate_axes)
    kept = [a for a in candidate_axes if not spread[a]["degenerate"]]
    keep_idx = [candidate_axes.index(a) for a in kept]
    clusters = dg.spearman_clusters(
        H_full[:, keep_idx], kept, threshold=redundancy_threshold, priority=axis_priority
    )
    representatives = select_balanced_axes(
        clusters["representatives"], axis_priority, max_per_tail=max_per_tail
    )
    return {"representatives": representatives, "spread": spread, "clusters": clusters["clusters"]}


def normalization_qc(
    H_sel: np.ndarray,
    chosen_axes: Sequence[str],
    *,
    lo_pct: float = ss.ROBUST_LO_PCT,
    hi_pct: float = ss.ROBUST_HI_PCT,
) -> dict:
    """Build-QC of the absolute-space normalization: bounds and clipped mass.

    The campaign normalization scales each axis by robust percentile bounds
    (rationale at :data:`scengen.subsample.ROBUST_LO_PCT`); pool members outside
    the bounds clip to the box faces. This reports, per chosen axis, the bounds
    actually used and the clipped fraction on each side, so the (bounded) face
    atoms are measured rather than assumed.

    Args:
        H_sel: ``(M, d)`` pool sub-image on the chosen axes (raw metric values).
        chosen_axes: Length-``d`` axis names (columns of ``H_sel``).
        lo_pct, hi_pct: Percentile bounds — must match the selector's.

    Returns:
        JSON-serializable dict with ``lo_pct``/``hi_pct`` and a per-axis map
        ``{axis: {lo, hi, clipped_low_frac, clipped_high_frac}}``.
    """
    H_sel = np.asarray(H_sel, dtype=float)
    lo, hi = ss.robust_range_bounds(H_sel, lo_pct=lo_pct, hi_pct=hi_pct)
    axes = {}
    for a, name in enumerate(chosen_axes):
        col = H_sel[:, a]
        axes[str(name)] = {
            "lo": float(lo[a]),
            "hi": float(hi[a]),
            "clipped_low_frac": float(np.mean(col < lo[a])),
            "clipped_high_frac": float(np.mean(col > hi[a])),
        }
    return {"lo_pct": float(lo_pct), "hi_pct": float(hi_pct), "axes": axes}


def select_from_candidate_image(
    H_candidates: np.ndarray,
    candidate_axes: Sequence[str],
    n: int,
    *,
    seed: int,
    selector_space: str,
    redundancy_threshold: float = 0.7,
    axis_priority: Sequence[str] = DEFAULT_AXIS_PRIORITY,
    max_per_tail: int = 2,
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
        selector_space: ``"abs"`` = absolute, range-scaled magnitude space — the CAMPAIGN selector,
            distorted by design (severe corners over-represented relative to their pool frequency).
            ``"cdf"`` = empirical-CDF/rank space — the retained NON-CAMPAIGN sensitivity, which
            preserves the pool marginals and distorts only the joint dependence among axes.
            Required: the two are different designs, so it is never defaulted.
        redundancy_threshold: Spearman ``|rho|`` cut for the redundancy clustering.
        axis_priority: Operational preference order for cluster representatives.
        max_per_tail: Maximum retained axes per hazard tail (dry / wet).
        selector_kwargs: Extra kwargs forwarded to the selector (e.g. ``k_pool``; for ``"abs"``,
            ``lo_pct``/``hi_pct`` override the robust p1/p99 campaign bounds and propagate to the
            coverage and normalization QC so all three stay consistent).

    Returns:
        Dict with ``selected_rows`` (sorted indices), ``chosen_axes`` (the screened axis set used for
        selection), ``candidate_axes``, ``H_candidates``, ``screen`` (spread + clusters),
        ``coverage`` (:func:`coverage_qc` on the chosen sub-image), and ``normalization``
        (:func:`normalization_qc`: the absolute-space bounds used and per-axis clipped fractions).
    """
    H_full = np.asarray(H_candidates, dtype=float)
    candidate_axes = list(candidate_axes)
    screen = screen_axes(
        H_full, candidate_axes, redundancy_threshold=redundancy_threshold,
        axis_priority=axis_priority, max_per_tail=max_per_tail,
    )
    chosen_axes = screen["representatives"]
    chosen_idx = [candidate_axes.index(a) for a in chosen_axes]
    H_sel = H_full[:, chosen_idx]

    # Selection space: "abs" (campaign; absolute robust range-scaled magnitude) or "cdf"
    # (non-campaign sensitivity; rank). Coverage QC is reported in BOTH geometries: each selector
    # fills its own geometry, so the cross-geometry entry is what exposes the trade the choice makes.
    kwargs = dict(selector_kwargs or {})
    lo_pct = float(kwargs.get("lo_pct", ss.ROBUST_LO_PCT))
    hi_pct = float(kwargs.get("hi_pct", ss.ROBUST_HI_PCT))
    if selector_space == "abs":
        sel = ss.absolute_filling_subsample(H_sel, n, seed=seed, **kwargs)
    elif selector_space == "cdf":
        kwargs.pop("lo_pct", None)
        kwargs.pop("hi_pct", None)
        sel = ss.cdf_filling_subsample(H_sel, n, seed=seed, **kwargs)
    else:
        raise ValueError(f"unknown selector_space={selector_space!r}")
    coverage = coverage_qc(H_sel, sel, seed=seed, lo_pct=lo_pct, hi_pct=hi_pct)
    normalization = normalization_qc(H_sel, chosen_axes, lo_pct=lo_pct, hi_pct=hi_pct)
    return {
        "selected_rows": sel,
        "chosen_axes": chosen_axes,
        "candidate_axes": candidate_axes,
        "H_candidates": H_full,
        "screen": {"spread": screen["spread"], "clusters": screen["clusters"], "representatives": chosen_axes},
        "coverage": coverage,
        "normalization": normalization,
    }
