"""Hazard-filling design driver (``scenario_design_methods.md`` §4.3).

A hazard-filling design owns a **candidate pool** of realizations and selects a
subset of them. It must select rather than generate: hazard coordinates (drought
magnitude, flood peak discharge, ...) are *emergent* properties of a
realized flow sequence, so no generator can be asked to produce a realization at
a prescribed drought severity. The Latin-hypercube anchors of the design are
therefore snapped to the nearest real pool member (see :mod:`scengen.subsample`).
Input-space stratification has no such constraint -- forcing parameters are a
knob on the generator -- and so is a generation design, not a subsample.

The selector is deterministic given its seed: LHS + nearest-neighbor snap. No
annealing, no discrepancy objective.

Live pipeline::

    candidate hazard image (streamed to hazard_image.npz at pool generation)
        ->  axis screen (degenerate drop + near-duplicate dedupe)
        ->  LHS + nearest-neighbor selection  ->  selected rows

The axis screen keeps **all non-degenerate hazard descriptors**: it drops only
axes with near-zero spread and, at ``|rho_S| >= 0.95``, prunes near-duplicate
groups to one canonical member so a single hazard concept cannot enter the
Euclidean snap twice under two names. Correlated-but-distinct descriptors are
deliberately retained — the design's coverage guarantee is per-axis marginal
(LHS stratifies every axis regardless of dimension), so extra axes cost nothing
in that guarantee. The Spearman correlation structure is reported as a
diagnostic (:func:`scengen.diagnostics.spearman_clusters`), never used to
reduce the axis set further.

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

#: Canonical priority order for choosing the ONE surviving member of a
#: near-duplicate axis group (|rho_S| >= threshold on the pool image). Members
#: of such a group are statistically interchangeable for coverage; this keeps
#: the operationally-preferred one. Per tail: the integrated-magnitude axis
#: first, then duration/intensity, then the rate facets.
DEFAULT_AXIS_PRIORITY: tuple[str, ...] = (
    "drought_magnitude", "drought_duration", "drought_severity",
    "drought_onset_rate", "drought_recovery_rate",
    "flood_peak_discharge", "flood_pulse_duration", "flood_rise_rate",
)

#: Spearman |rho| at or above which two axes are near-duplicates of one hazard
#: concept and are pruned to the single canonical member. Deliberately high:
#: correlated-but-distinct descriptors below it are all retained.
DEDUPE_RHO_THRESHOLD: float = 0.95


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


def screen_hazard_axes(
    H_candidates: np.ndarray,
    candidate_axes: Sequence[str],
    *,
    dedupe_threshold: float = DEDUPE_RHO_THRESHOLD,
    axis_priority: Sequence[str] = DEFAULT_AXIS_PRIORITY,
) -> dict:
    """Axis screen: keep all non-degenerate hazard descriptors minus near-duplicates.

    Drops (a) degenerate axes (near-zero spread; see
    :func:`scengen.diagnostics.per_metric_spread`) and (b) near-duplicates —
    axes connected by ``|rho_S| >= dedupe_threshold`` on the pool image form a
    group pruned to its single highest-``axis_priority`` member, so one hazard
    concept cannot enter the Euclidean snap twice under two names. Everything
    else is retained: no clustering-based reduction, no per-tail cap. Shared by
    the live selection driver (:func:`select_from_candidate_image`) and the
    selector diagnostics, so both always screen identically.

    Args:
        H_candidates: ``(M, n_candidates)`` candidate hazard image.
        candidate_axes: Length-``n_candidates`` axis names.
        dedupe_threshold: Spearman ``|rho|`` at or above which two axes are
            near-duplicates (groups are the connected components of that
            relation).
        axis_priority: Canonical preference order for the surviving member of a
            near-duplicate group (unknown names rank last, in pool order).

    Returns:
        JSON-serializable dict with ``retained`` (axis names in candidate
        order), ``dropped`` (``{axis: {reason, kept_member?, rho_with_kept?}}``),
        ``near_duplicate_groups``, ``spearman_axes`` / ``spearman_rho`` (the
        full matrix over the non-degenerate axes), ``dedupe_threshold``, and
        ``spread`` (per-axis degeneracy stats).

    Raises:
        ValueError: If fewer than 3 axes are retained (a pathologically
            degenerate or redundant pool).
    """
    from scipy.stats import spearmanr

    H_full = np.asarray(H_candidates, dtype=float)
    candidate_axes = list(candidate_axes)
    spread = dg.per_metric_spread(H_full, candidate_axes)
    dropped: dict[str, dict] = {
        a: {"reason": "degenerate"} for a in candidate_axes if spread[a]["degenerate"]
    }
    kept = [a for a in candidate_axes if a not in dropped]
    keep_idx = [candidate_axes.index(a) for a in kept]

    rho = np.eye(len(kept))
    if len(kept) == 2:  # spearmanr returns a scalar for two columns
        r, _ = spearmanr(H_full[:, keep_idx[0]], H_full[:, keep_idx[1]])
        rho = np.array([[1.0, float(r)], [float(r), 1.0]])
    elif len(kept) > 2:
        rho_raw, _ = spearmanr(H_full[:, keep_idx])
        rho = np.atleast_2d(rho_raw)

    # Near-duplicate groups: connected components of |rho_S| >= threshold.
    parent = list(range(len(kept)))

    def _find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(kept)):
        for j in range(i + 1, len(kept)):
            if abs(rho[i, j]) >= dedupe_threshold:
                parent[_find(i)] = _find(j)

    rank = {a: r for r, a in enumerate(axis_priority)}
    groups: dict[int, list[int]] = {}
    for i in range(len(kept)):
        groups.setdefault(_find(i), []).append(i)

    near_duplicate_groups: list[list[str]] = []
    for members in groups.values():
        if len(members) == 1:
            continue
        names = [kept[i] for i in members]
        near_duplicate_groups.append(names)
        survivor = min(names, key=lambda a: (rank.get(a, len(rank)), kept.index(a)))
        for name in names:
            if name != survivor:
                dropped[name] = {
                    "reason": "near_duplicate",
                    "kept_member": survivor,
                    "rho_with_kept": float(rho[kept.index(name), kept.index(survivor)]),
                }

    retained = [a for a in candidate_axes if a not in dropped]
    if len(retained) < 3:
        raise ValueError(
            f"Axis screen retained only {len(retained)} hazard axes "
            f"({retained}) of candidates {candidate_axes} at "
            f"|rho_S| >= {dedupe_threshold}; at least 3 are required. The pool "
            f"is pathologically degenerate or redundant — inspect the candidate "
            f"hazard image."
        )
    return {
        "retained": retained,
        "dropped": dropped,
        "near_duplicate_groups": near_duplicate_groups,
        "spearman_axes": kept,
        "spearman_rho": np.asarray(rho, dtype=float).tolist(),
        "dedupe_threshold": float(dedupe_threshold),
        "spread": spread,
    }


def normalization_qc(
    H_sel: np.ndarray,
    chosen_axes: Sequence[str],
    *,
    lo_pct: float = ss.ROBUST_LO_PCT,
    hi_pct: float = ss.ROBUST_HI_PCT,
    selected_rows: np.ndarray | None = None,
) -> dict:
    """Build-QC of the absolute-space normalization: bounds and clipped mass.

    The campaign normalization scales each axis by robust percentile bounds
    (rationale at :data:`scengen.subsample.ROBUST_LO_PCT`); pool members outside
    the bounds clip to the box faces. This reports, per chosen axis, the bounds
    actually used and the clipped fraction on each side, so the (bounded) face
    atoms are measured rather than assumed. When ``selected_rows`` is given, the
    share of SELECTED members sitting on at least one box face (any coordinate
    outside its bounds) is reported too — face-resident members have one or more
    degenerate (clipped) coordinates in the selection geometry.

    Args:
        H_sel: ``(M, d)`` pool sub-image on the chosen axes (raw metric values).
        chosen_axes: Length-``d`` axis names (columns of ``H_sel``).
        lo_pct, hi_pct: Percentile bounds — must match the selector's.
        selected_rows: Row indices of the selected set into ``H_sel``.

    Returns:
        JSON-serializable dict with ``lo_pct``/``hi_pct``, a per-axis map
        ``{axis: {lo, hi, clipped_low_frac, clipped_high_frac}}``, and (when
        ``selected_rows`` is given) ``selected_face_resident_frac``.
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
    out = {"lo_pct": float(lo_pct), "hi_pct": float(hi_pct), "axes": axes}
    if selected_rows is not None:
        sel = H_sel[np.asarray(selected_rows, dtype=int)]
        on_face = ((sel < lo) | (sel > hi)).any(axis=1)
        out["selected_face_resident_frac"] = float(np.mean(on_face))
    return out


def select_from_candidate_image(
    H_candidates: np.ndarray,
    candidate_axes: Sequence[str],
    n: int,
    *,
    seed: int,
    selector_space: str,
    selection_axes: Sequence[str] | None = None,
    dedupe_threshold: float = DEDUPE_RHO_THRESHOLD,
    axis_priority: Sequence[str] = DEFAULT_AXIS_PRIORITY,
    selector_kwargs: dict | None = None,
) -> dict:
    """Screen a precomputed candidate hazard image, then space-fill-select ``n`` scenarios.

    The single entry point of the hazard-filling design. Given the candidate image ``H_candidates``
    (``M x n_candidates``) already computed for a candidate pool — streamed at pool generation and
    reloaded from ``hazard_image.npz`` — it runs the axis screen
    (:func:`screen_hazard_axes`: drop degenerate axes; prune near-duplicate groups at
    ``|rho_S| >= dedupe_threshold`` to one canonical member; retain everything else) and then the
    LHS + nearest-neighbor selector on the screened sub-image. The axis set is screened per pool
    rather than hard-coded, so it adapts if the generator or pool changes.

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
        selection_axes: Optional subset of ``candidate_axes`` to consider for selection; the
            screen and the selector then operate on this restriction (the remaining image
            columns stay computed and reportable, they just do not enter the snap distance).
            The caller owns this policy choice — e.g. a campaign axis set fixed by a
            pool-size saturation diagnostic when the full set cannot meet its per-axis
            adequacy gate at an affordable pool size. Default ``None`` = all candidates.
        dedupe_threshold: Spearman ``|rho|`` at or above which two axes are near-duplicates.
        axis_priority: Canonical preference order for the surviving member of a near-duplicate
            group.
        selector_kwargs: Extra kwargs forwarded to the selector (e.g. ``k_pool``; for ``"abs"``,
            ``lo_pct``/``hi_pct`` override the robust p1/p99 campaign bounds and propagate to the
            coverage and normalization QC so all three stay consistent).

    Returns:
        Dict with ``selected_rows`` (sorted indices), ``chosen_axes`` (the screened axis set used for
        selection), ``candidate_axes``, ``H_candidates``, ``screen`` (the full
        :func:`screen_hazard_axes` output: retained/dropped axes with reasons, Spearman matrix,
        threshold, spread), ``coverage`` (:func:`coverage_qc` on the chosen sub-image), and
        ``normalization`` (:func:`normalization_qc`: the absolute-space bounds used and per-axis
        clipped fractions).
    """
    H_full = np.asarray(H_candidates, dtype=float)
    candidate_axes = list(candidate_axes)
    if selection_axes is not None:
        missing = [a for a in selection_axes if a not in candidate_axes]
        if missing:
            raise ValueError(
                f"selection_axes not in candidate_axes: {missing} (candidates: {candidate_axes})"
            )
        keep = [candidate_axes.index(a) for a in selection_axes]
        H_full = H_full[:, keep]
        candidate_axes = list(selection_axes)
    screen = screen_hazard_axes(
        H_full, candidate_axes, dedupe_threshold=dedupe_threshold, axis_priority=axis_priority,
    )
    chosen_axes = screen["retained"]
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
    normalization = normalization_qc(
        H_sel, chosen_axes, lo_pct=lo_pct, hi_pct=hi_pct, selected_rows=sel
    )
    return {
        "selected_rows": sel,
        "chosen_axes": chosen_axes,
        "candidate_axes": candidate_axes,
        "H_candidates": H_full,
        "screen": screen,
        "coverage": coverage,
        "normalization": normalization,
    }
