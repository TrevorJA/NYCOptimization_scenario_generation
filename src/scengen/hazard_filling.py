"""Hazard-filling design driver (methods 4.6).

Composes the pieces of the hazard-filling scenario ensemble:

    staged master pool  ->  monthly aggregate NYC inflow  ->  hazard image H
                        ->  stratified-maximin subsample  ->  subset + manifest

The subset of realization indices + a provenance manifest are written to the
shared staging directory; NYCOptimization's ``hazard_filling`` design reads the
manifest and runs the optimizer on the selected realizations.

Flexibility (both expected to change later):
  - Original ensemble subsampled from: any staged ``kn_{Y}yr_n{N}`` pool (passed
    in); the initial draft uses stationary Kirsch-Nowak 5-year records.
  - Hazard metrics: ``metric_names`` selects the axes (default the MOEA-FIND
    "primary" SSI set, see :mod:`scengen.hazard_metrics`).

Pure-array entry point :func:`build_hazard_filling_subset` has no pywrdrb/HDF5
dependency and is unit-tested directly. :func:`load_ensemble_monthly_aggregate`
reads a staged SynHydro ensemble HDF5.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from . import hazard_metrics as hm
from . import subsample as ss
from .manifest import EnsembleManifest


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


def build_hazard_filling_subset(
    scenario_monthly: np.ndarray,
    reference_monthly: np.ndarray,
    n: int,
    *,
    seed: int,
    metric_names: Sequence[str] = hm.PRIMARY_METRICS,
    timescale: int = 3,
    dist: str = "gamma",
    reference_start: str = "1945-10-01",
    selector_kwargs: dict | None = None,
) -> dict:
    """Compute the hazard image and select a space-filling subset of ``n`` scenarios.

    Args:
        scenario_monthly: ``(n_scenarios, n_months)`` monthly aggregate inflow of
            the master pool.
        reference_monthly: 1D historical monthly aggregate inflow for the SSI fit.
        n: Number of scenarios to select.
        seed: Selector RNG seed.
        metric_names: Hazard axes (default the "primary" SSI set).
        timescale, dist, reference_start: SSI configuration.
        selector_kwargs: Extra kwargs for ``hazard_filling_subsample``.

    Returns:
        Dict with ``selected_rows`` (indices into ``scenario_monthly``),
        ``hazard_axes``, ``H`` (full hazard image), and ``coverage`` (QC: the
        selected subset's discrepancy vs a random subset of the same size).
    """
    H, names = hm.compute_hazard_image(
        scenario_monthly, reference_monthly,
        metric_names=metric_names, timescale=timescale, dist=dist,
        reference_start=reference_start,
    )
    sel = ss.hazard_filling_subsample(H, n, seed=seed, **(selector_kwargs or {}))
    X = ss.empirical_cdf_normalize(H)
    lb, ub = np.zeros(X.shape[1]), np.ones(X.shape[1])
    rand = ss.random_subsample(H, n, seed=seed)
    coverage = {
        "selected_L2_star": ss.coverage_metrics(X[sel], lb, ub)["L2_star_discrepancy"],
        "random_L2_star": ss.coverage_metrics(X[rand], lb, ub)["L2_star_discrepancy"],
    }
    return {
        "selected_rows": sel,
        "hazard_axes": names,
        "H": H,
        "coverage": coverage,
    }


def write_subset_manifest(
    path: str | Path,
    *,
    master_slug: str,
    selected_global_indices: Sequence[int],
    hazard_axes: Sequence[str],
    realization_years: int,
    seed: int,
    coverage: dict | None = None,
    master_seed: int = 0,
    created: str = "",
    notes: str = "",
) -> Path:
    """Write the hazard-filling subset manifest (the artifact NYCOptimization reads)."""
    manifest = EnsembleManifest(
        design="hazard_filling",
        draw=0,
        n_realizations=len(selected_global_indices),
        realization_years=realization_years,
        master_seed=master_seed,
        realization_global_indices=list(selected_global_indices),
        forcing_hash="stationary",
        hazard_axes=list(hazard_axes),
        screen_result={
            "metric_set": list(hazard_axes),
            "selector": "sa_stratified_maximin",
            "selector_seed": seed,
            "coverage": coverage or {},
        },
        slug=master_slug,
        created=created,
        source_kind="synhydro_kn",
        notes=notes,
    )
    return manifest.write(Path(path).parent, filename=Path(path).name)
