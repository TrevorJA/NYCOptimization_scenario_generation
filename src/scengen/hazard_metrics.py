"""Hazard metrics for subsampling (methods 3.3).

Computes the per-scenario hazard vector ``h(s)`` used by the hazard-filling
selector. The default axis set is MOEA-FIND's production **"primary"** drought
set -- ``mean_severity``, ``mean_magnitude``, ``time_in_drought_fraction`` --
computed from the Standardized Streamflow Index (SSI-3) of each scenario's
monthly aggregate NYC inflow. The full 10-metric SSI Tier-A set is available so
the axis set is easy to change later (flexibility requirement).

The SSI Tier-A extraction and the flow->series / SSI-calculator helpers are
COPIED (not imported) from MOEA-FIND (``src/metrics/ssi_common.py`` and
``src/metrics/objectives.py``) so this repo has no dependency on that repo. SSI
itself is provided by **SynHydro** (``synhydro.droughts.ssi``), an allowed
dependency, imported lazily so the module imports without it.

Design choices (per MOEA-FIND DD-11 and methods 3.3):
  - The SSI gamma distribution is fit ONCE on the historical reference record
    (per calendar month), then applied to every scenario, so hazard
    coordinates are comparable across scenarios.
  - Series are water-year aligned (October start) so SSI's per-calendar-month
    fit applies to the correct months of each scenario.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

#: SSI value at/below which a month counts as drought-stressed (SynHydro's
#: "critical drought" convention; matches MOEA-FIND TIME_IN_DROUGHT_THRESHOLD).
TIME_IN_DROUGHT_THRESHOLD: float = -1.0

#: All SSI Tier-A event metrics available (copied from MOEA-FIND ssi_common).
AVAILABLE_METRICS: tuple[str, ...] = (
    "frequency", "mean_duration", "max_duration",
    "mean_magnitude", "max_magnitude",
    "mean_severity", "worst_severity", "mean_avg_severity",
    "time_in_drought_fraction",
)

#: MOEA-FIND production "primary" set (drought_metrics.PRESETS["primary"]).
PRIMARY_METRICS: tuple[str, ...] = (
    "mean_severity", "mean_magnitude", "time_in_drought_fraction",
)

#: Default pywrdrb nodes summed to form the aggregate NYC inflow series.
DEFAULT_NYC_INFLOW_NODES: tuple[str, ...] = ("cannonsville", "pepacton", "neversink")

_WATER_YEAR_START = "2000-10-01"  # October start; only month-of-year matters


def flows_to_series(
    monthly_flows: np.ndarray, start_date: str = _WATER_YEAR_START, freq: str = "MS"
) -> pd.Series:
    """Wrap a 1D/2D monthly-flow array in a pd.Series with a DatetimeIndex.

    Copied from MOEA-FIND ``objectives.flows_to_series``. SynHydro's SSI requires
    a DatetimeIndex; ``start_date`` should be an October so the per-calendar-month
    SSI fit aligns with each scenario's water-year months.
    """
    arr = np.asarray(monthly_flows, dtype=float)
    if arr.ndim == 2:
        arr = arr.flatten()
    index = pd.date_range(start=start_date, periods=len(arr), freq=freq)
    return pd.Series(arr, index=index, name="flow_mgd")


def make_ssi_calculator(timescale: int = 3, dist: str = "gamma", fit_freq: str = "M"):
    """Create a configured (unfitted) SynHydro SSI calculator.

    Copied from MOEA-FIND ``objectives.make_ssi_calculator``. SynHydro is imported
    lazily so this module imports without it.
    """
    from synhydro.droughts.ssi import SSI  # lazy: allowed dependency
    return SSI(dist=dist, timescale=timescale, fit_freq=fit_freq)


def fit_reference_ssi(
    reference_monthly: np.ndarray,
    *,
    timescale: int = 3,
    dist: str = "gamma",
    start_date: str = "1945-10-01",
):
    """Fit an SSI calculator on the historical reference monthly series.

    Args:
        reference_monthly: 1D monthly aggregate-inflow array for the historical
            reference record (water-year aligned, October start).
        timescale: SSI accumulation period in months (SSI-3 default).
        dist: Fitting distribution (gamma default).
        start_date: October-aligned start for the reference index.

    Returns:
        A fitted SynHydro SSI calculator to transform scenarios with.
    """
    calc = make_ssi_calculator(timescale=timescale, dist=dist)
    calc.fit(flows_to_series(reference_monthly, start_date=start_date))
    return calc


def _tier_a_metrics(ssi_series: pd.Series, dm: pd.DataFrame | None = None) -> dict:
    """The SSI Tier-A event metrics from an SSI series (copied from MOEA-FIND ssi_common)."""
    from synhydro.droughts.ssi import get_drought_metrics  # lazy
    if dm is None:
        dm = get_drought_metrics(ssi_series)
    valid = ssi_series.dropna()
    n_valid = len(valid)
    n_years = n_valid / 12.0 if n_valid > 0 else 0.0
    time_in_drought = (
        float((valid <= TIME_IN_DROUGHT_THRESHOLD).sum() / n_valid) if n_valid > 0 else 0.0
    )
    if len(dm) == 0:
        out = {k: 0.0 for k in AVAILABLE_METRICS}
        out["time_in_drought_fraction"] = time_in_drought
        return out
    return {
        "frequency": float(len(dm) / n_years * 10) if n_years > 0 else 0.0,
        "mean_duration": float(dm["duration"].mean()),
        "max_duration": float(dm["duration"].max()),
        "mean_magnitude": float(dm["magnitude"].abs().mean()),
        "max_magnitude": float(dm["magnitude"].abs().max()),
        "mean_severity": float(dm["severity"].abs().mean()),
        "worst_severity": float(dm["severity"].abs().max()),
        "mean_avg_severity": float(dm["avg_severity"].abs().mean()),
        "time_in_drought_fraction": time_in_drought,
    }


def compute_metrics_for_scenario(
    monthly_flows: np.ndarray,
    calc,
    *,
    metric_names: Sequence[str] = PRIMARY_METRICS,
    start_date: str = _WATER_YEAR_START,
) -> dict:
    """Hazard metrics for one scenario's monthly aggregate-inflow series.

    Args:
        monthly_flows: 1D monthly aggregate inflow for the scenario (Oct start).
        calc: A fitted SSI calculator from :func:`fit_reference_ssi`.
        metric_names: Subset of :data:`AVAILABLE_METRICS` to return.
        start_date: October-aligned start so per-calendar-month SSI applies.

    Returns:
        Dict of the requested metric values.
    """
    ssi = calc.transform(flows_to_series(monthly_flows, start_date=start_date))
    all_metrics = _tier_a_metrics(ssi)
    return {m: all_metrics[m] for m in metric_names}


def compute_hazard_image(
    scenario_monthly: np.ndarray,
    reference_monthly: np.ndarray,
    *,
    metric_names: Sequence[str] = PRIMARY_METRICS,
    timescale: int = 3,
    dist: str = "gamma",
    reference_start: str = "1945-10-01",
) -> tuple[np.ndarray, list[str]]:
    """Compute the hazard image ``H`` for an ensemble of scenarios.

    Fits the SSI distribution once on ``reference_monthly`` and transforms each
    scenario row, returning the ``(n_scenarios, n_metrics)`` hazard image.

    Args:
        scenario_monthly: ``(n_scenarios, n_months)`` monthly aggregate inflow.
        reference_monthly: 1D historical monthly aggregate inflow for the SSI fit.
        metric_names: Hazard axes to compute (default the "primary" set).
        timescale, dist: SSI configuration.
        reference_start: October-aligned start for the reference fit.

    Returns:
        ``(H, list(metric_names))``.
    """
    calc = fit_reference_ssi(
        reference_monthly, timescale=timescale, dist=dist, start_date=reference_start
    )
    names = list(metric_names)
    rows = [
        [compute_metrics_for_scenario(row, calc, metric_names=names)[m] for m in names]
        for row in np.asarray(scenario_monthly, dtype=float)
    ]
    return np.asarray(rows, dtype=float), names
