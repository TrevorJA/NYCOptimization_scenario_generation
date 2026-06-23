"""Hazard-filling design driver (methods 4.6).

Composes the pieces of the hazard-filling scenario ensemble:

    staged master pool  ->  monthly aggregate NYC inflow  ->  hazard image H
                        ->  LHS+nearest-neighbor subsample  ->  final ensemble HDF5

The generator emits the **final ensemble** that the optimizer consumes directly:
the selected realizations are sliced out of the staged master pool, renumbered
``0..N-1``, and written as a standalone pywrdrb-format ensemble (``gage_flow``
and ``catchment_inflow`` HDF5s) plus an informational ``_meta.json`` provenance
sidecar. There is no manifest-as-contract and no per-realization index override
on the optimizer side -- NYCOptimization resolves the staged ensemble by slug
like any other.

Flexibility (both expected to change later):
  - Original ensemble subsampled from: any staged ``kn_{Y}yr_n{N}`` pool (passed
    in); the initial draft uses stationary Kirsch-Nowak 5-year records.
  - Hazard metrics: ``metric_names`` selects the axes (default the MOEA-FIND
    "primary" SSI set, see :mod:`scengen.hazard_metrics`).

Pure-array entry point :func:`build_hazard_filling_subset` has no pywrdrb/HDF5
dependency and is unit-tested directly. :func:`load_ensemble_monthly_aggregate`
and :func:`stage_subset_ensemble` read/write staged SynHydro ensemble HDF5s.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from . import diagnostics as dg
from . import hazard_metrics as hm
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


def load_ensemble_monthly_aggregate(
    catchment_inflow_hdf5: str | Path,
    nodes: Sequence[str] = hm.DEFAULT_NYC_INFLOW_NODES,
    *,
    agg: str = "mean",
) -> tuple[np.ndarray, list[int]]:
    """Load a staged ensemble and build the monthly aggregate NYC inflow per realization.

    Args:
        catchment_inflow_hdf5: Path to a staged ``catchment_inflow_mgd.hdf5``
            (SynHydro by-node format).
        nodes: Nodes summed to form the aggregate inflow (default the three NYC
            reservoir catchments).
        agg: Daily->monthly aggregation ("mean" MGD or "sum").

    Returns:
        ``(monthly_matrix, realization_ids)`` where ``monthly_matrix`` is
        ``(n_realizations, n_months)`` and rows align with ``realization_ids``.
    """
    from synhydro.core.ensemble import Ensemble  # lazy: allowed dependency

    ens = Ensemble.from_hdf5(str(catchment_inflow_hdf5), stored_by_node=True)
    realization_ids = sorted(ens.data_by_realization)
    rows = []
    for rid in realization_ids:
        df = ens.data_by_realization[rid]
        missing = [n for n in nodes if n not in df.columns]
        if missing:
            raise KeyError(f"nodes {missing} not in ensemble columns {list(df.columns)}")
        agg_daily = df.loc[:, list(nodes)].sum(axis=1)
        rows.append(daily_to_monthly(agg_daily, agg=agg))
    n_months = min(len(r) for r in rows)
    monthly = np.vstack([r[:n_months] for r in rows])
    return monthly, list(realization_ids)


def load_ensemble_daily_aggregate(
    catchment_inflow_hdf5: str | Path,
    nodes: Sequence[str] = hm.DEFAULT_NYC_INFLOW_NODES,
) -> tuple[np.ndarray, list[int]]:
    """Load a staged ensemble and build the DAILY aggregate NYC inflow per realization.

    Used for the wet (flood) hazard axes, which need daily resolution. Sums the
    ``nodes`` columns per day (no temporal aggregation).

    Args:
        catchment_inflow_hdf5: Path to a staged ``catchment_inflow_mgd.hdf5``.
        nodes: Nodes summed to form the aggregate inflow (default the three NYC
            reservoir catchments).

    Returns:
        ``(daily_matrix, realization_ids)`` where ``daily_matrix`` is
        ``(n_realizations, n_days)`` and rows align with ``realization_ids``.
    """
    from synhydro.core.ensemble import Ensemble  # lazy: allowed dependency

    ens = Ensemble.from_hdf5(str(catchment_inflow_hdf5), stored_by_node=True)
    realization_ids = sorted(ens.data_by_realization)
    rows = []
    for rid in realization_ids:
        df = ens.data_by_realization[rid]
        missing = [n for n in nodes if n not in df.columns]
        if missing:
            raise KeyError(f"nodes {missing} not in ensemble columns {list(df.columns)}")
        rows.append(df.loc[:, list(nodes)].sum(axis=1).to_numpy(dtype=float))
    n_days = min(len(r) for r in rows)
    daily = np.vstack([r[:n_days] for r in rows])
    return daily, list(realization_ids)


def build_hazard_filling_subset(
    scenario_monthly: np.ndarray,
    scenario_daily: np.ndarray,
    reference_monthly: np.ndarray,
    reference_daily: np.ndarray,
    n: int,
    *,
    seed: int,
    dry_timescale: int = 6,
    flood_threshold_pct: float = 95.0,
    redundancy_threshold: float = 0.7,
    axis_priority: Sequence[str] = DEFAULT_AXIS_PRIORITY,
    max_per_tail: int = 2,
    selector_space: str = "cdf",
    dist: str = "gamma",
    reference_start: str = "1945-10-01",
    selector_kwargs: dict | None = None,
) -> dict:
    """Screen the candidate hazard axes per pool, then space-fill-select ``n`` scenarios.

    Pipeline (methods 3.3 + 4.6): compute the 6-candidate wet+dry event-descriptor
    hazard image, screen it (Olden & Poff: drop degenerate axes, then keep one
    operationally-preferred representative per ``|rho_S| >= redundancy_threshold``
    cluster), and run the LHS+nearest-neighbor selector on the screened sub-image.
    The axis set is chosen per master pool rather than hard-coded, so it adapts if
    the generator/pool changes.

    Args:
        scenario_monthly: ``(M, n_months)`` monthly aggregate NYC inflow.
        scenario_daily: ``(M, n_days)`` daily aggregate NYC inflow (same scenarios).
        reference_monthly: 1D historical monthly aggregate inflow (dry SSI fit).
        reference_daily: 1D historical daily aggregate inflow (POT threshold + mean).
        n: Number of scenarios to select.
        seed: Selector (LHS) RNG seed.
        dry_timescale: Drought SSI accumulation in months (SSI-6 default).
        flood_threshold_pct: POT high-flow threshold percentile (95 = Q5).
        redundancy_threshold: Spearman |rho| cut for the redundancy clustering.
        axis_priority: Operational preference order for cluster representatives.
        dist, reference_start: SSI configuration.
        selector_kwargs: Extra kwargs for ``hazard_filling_subsample``.

    Returns:
        Dict with ``selected_rows`` (sorted indices), ``chosen_axes`` (the
        screened axis set used for selection), ``candidate_axes`` (all 6),
        ``H_candidates`` (M x 6), ``screen`` (spread + clusters), and ``coverage``
        (selected vs random L2-star on the chosen sub-image).
    """
    H_full, candidate_axes = hm.compute_candidate_hazard_image(
        scenario_monthly, scenario_daily, reference_monthly, reference_daily,
        dry_timescale=dry_timescale, flood_threshold_pct=flood_threshold_pct,
        dist=dist, reference_start=reference_start,
    )

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

    # Selection space: "cdf" (faithful, rank space) or "abs" (distorted, absolute
    # magnitude space). Coverage QC is always reported in CDF/rank space so the
    # two arms are comparable on the same discrepancy scale.
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
        "screen": {
            "spread": spread,
            "clusters": clusters["clusters"],
            "representatives": chosen_axes,
        },
        "coverage": coverage,
    }


def stage_subset_ensemble(
    pool_dir: str | Path,
    out_dir: str | Path,
    selected_source_ids: Sequence[int],
    *,
    meta: dict,
    files: Sequence[str] = ("gage_flow_mgd.hdf5", "catchment_inflow_mgd.hdf5"),
) -> Path:
    """Write the final hazard-filling ensemble by slicing the master pool.

    Reads each pywrdrb-format HDF5 in ``pool_dir``, keeps only the
    ``selected_source_ids`` realizations, renumbers them ``0..N-1``, and writes
    the reduced ensemble to ``out_dir`` together with an informational
    ``_meta.json`` provenance sidecar. The reduced HDF5s are what the optimizer
    loads directly (resolved by slug); the manifest/index-override path is gone.

    Args:
        pool_dir: Staged master-pool directory (holds the ``files`` HDF5s).
        out_dir: Destination directory for the final ensemble (created if absent).
        selected_source_ids: Pool realization ids to keep (the from_hdf5 enumerate
            keys, i.e. the ``realization_ids`` returned by
            :func:`load_ensemble_monthly_aggregate`). Output realizations are
            renumbered ``0..N-1`` in this order.
        meta: Provenance dict written verbatim to ``_meta.json``. Must carry at
            least ``n_realizations`` and ``realization_years`` (NYCOptimization
            resolves the staged ensemble from these).
        files: HDF5 basenames to slice (default the pywrdrb gage + catchment pair).

    Returns:
        The ``out_dir`` path.
    """
    from synhydro.core.ensemble import Ensemble  # lazy: allowed dependency

    pool_dir, out_dir = Path(pool_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    src_ids = [int(s) for s in selected_source_ids]

    for fname in files:
        src_path = pool_dir / fname
        if not src_path.exists():
            raise FileNotFoundError(f"pool file missing: {src_path}")
        pool = Ensemble.from_hdf5(str(src_path), stored_by_node=True).data_by_realization
        missing = [s for s in src_ids if s not in pool]
        if missing:
            raise KeyError(f"selected ids {missing} not in pool {src_path.name}")
        reduced = {new_id: pool[s] for new_id, s in enumerate(src_ids)}
        Ensemble(reduced).to_hdf5(str(out_dir / fname))

    (out_dir / "_meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True))
    return out_dir
