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


# ---------------------------------------------------------------------------
# Run-theory event descriptors (candidate hazard axes; asymmetric by tail)
# ---------------------------------------------------------------------------
#
# Per-event, NOT averaged across events. The two tails use DIFFERENT, physically
# faithful event models (per independent academic review): droughts are slow
# depletion -> SSI run theory (deficit below a standardized threshold); floods
# are fast pulses -> peaks-over-threshold (POT) on the daily series. A single
# symmetric standardized index (negated SSI) is NOT used for floods: a gamma-fit
# deficit index saturates the upper tail where floods live. Grounding:
# Yevjevich (1967) run theory; Vicente-Serrano et al. (2012) SSI; McKee et al.
# (1993) on accumulation timescale (SSI-6 for reservoir drought); the five
# flow-regime facets and high-pulse indices of Richter et al. (1996, IHA) /
# Olden & Poff (2003); Lang et al. (1999) POT/partial-duration series; Baker et
# al. (2004) flashiness; Brunner et al. (2021) joint flood-drought framing.

#: Dry (drought / low-flow) candidate axes: controlling-event SSI run-theory
#: descriptors on the monthly aggregate NYC inflow (SSI-6 reservoir timescale).
#: The magnitude/duration/intensity trio is collinear by construction (run
#: theory: deficit ~ duration x intensity); the rate-of-change descriptors
#: (onset/recovery; IHA Group 5; flash-drought lit Otkin 2018) are the
#: orthogonal facet that yields a genuine SECOND drought axis after screening.
DRY_EVENT_METRICS: tuple[str, ...] = (
    "drought_deficit_volume",  # |sum SSI over run| — magnitude facet
    "drought_duration",        # run length (months) — duration facet
    "drought_peak_depth",      # |min SSI in run| — intensity
    "drought_onset_rate",      # |peak SSI| / months to trough — rate of change
    "drought_recovery_rate",   # |peak SSI| / recovery months — rate of change
)

#: Wet (flood / high-flow) candidate axes: critical peaks-over-threshold pulse
#: descriptors on the daily aggregate NYC inflow (Q5 / 95th-pct threshold).
WET_EVENT_METRICS: tuple[str, ...] = (
    "flood_peak_magnitude",    # window max daily flow / reference mean — magnitude
    "flood_pulse_duration",    # days above threshold in the critical pulse — duration
    "flood_rise_rate",         # max 1-day rise of the critical rising limb / ref mean — rate
)

#: Full candidate pool screened down to the final low-redundancy axis set.
CANDIDATE_EVENT_METRICS: tuple[str, ...] = DRY_EVENT_METRICS + WET_EVENT_METRICS


def critical_event_descriptors(
    ssi_series: pd.Series, *, end_threshold: int = 3, select: str = "controlling"
) -> dict:
    """Run-theory descriptors of the single controlling drought event.

    Reuses SynHydro ``get_drought_metrics`` (Yevjevich run theory): an event is a
    run of ``SSI < 0`` reaching ``SSI <= -1``, terminated by ``end_threshold``
    consecutive non-negative steps. Per review, the default selector is
    ``"controlling"`` = the run with the largest cumulative deficit (the
    operationally binding event under fixed initial storage), which is more
    stable than the deepest-severity argmax.

    Args:
        ssi_series: SSI series (DatetimeIndex).
        end_threshold: Consecutive non-negative steps to terminate an event.
        select: ``"controlling"`` (max |cumulative deficit|; default),
            ``"deepest"`` (min severity), or ``"first"`` (earliest run).

    Returns:
        Dict with non-negative ``duration``, ``volume`` (|cumulative SSI|), and
        ``depth`` (|peak SSI|). All zero when the scenario has no critical event.
    """
    from synhydro.droughts.ssi import get_drought_metrics  # lazy

    dm = get_drought_metrics(ssi_series, end_drought_threshold_months=end_threshold)
    if len(dm) == 0:
        return {"duration": 0.0, "volume": 0.0, "depth": 0.0,
                "onset_rate": 0.0, "recovery_rate": 0.0}
    if select == "controlling":
        i = dm["magnitude"].astype(float).abs().idxmax()
    elif select == "deepest":
        i = dm["severity"].astype(float).idxmin()
    elif select == "first":
        i = dm.index[0]
    else:
        raise ValueError(f"unknown select={select!r}")
    crit = dm.loc[i]
    depth = float(abs(crit["severity"]))

    # Rate-of-change descriptors (orthogonal to magnitude/duration/intensity).
    start = pd.Timestamp(crit["start"])
    peak = pd.Timestamp(crit["max_severity_date"])
    months_to_peak = max(1, (peak.year - start.year) * 12 + (peak.month - start.month))
    recovery_months = max(1.0, float(crit.get("recovery_period", 0.0)))
    return {
        "duration": float(crit["duration"]),
        "volume": float(abs(crit["magnitude"])),
        "depth": depth,
        "onset_rate": depth / months_to_peak,
        "recovery_rate": depth / recovery_months,
    }


def pot_flood_descriptors(
    daily: np.ndarray, *, threshold: float, ref_mean: float
) -> dict:
    """Peaks-over-threshold descriptors of the critical (largest-peak) high-flow pulse.

    Floods are characterized by an asymmetric, peak-driven POT event model (Lang
    et al. 1999; IHA high-pulse indices, Richter et al. 1996), not a standardized
    index. The critical pulse is the above-``threshold`` run containing the
    window's maximum daily flow.

    Args:
        daily: 1D daily aggregate inflow for the scenario window.
        threshold: High-flow threshold (e.g. reference Q5 / 95th-pct daily flow).
        ref_mean: Reference mean daily flow, used to non-dimensionalize magnitudes.

    Returns:
        Dict with ``peak_magnitude`` (window max / ``ref_mean``; always defined,
        so no zero-inflation), ``pulse_duration`` (days above ``threshold`` in the
        critical pulse), and ``rise_rate`` (max 1-day rise on the critical rising
        limb / ``ref_mean``).
    """
    daily = np.asarray(daily, dtype=float)
    ref_mean = float(ref_mean) if ref_mean > 0 else 1.0
    peak_magnitude = float(daily.max() / ref_mean)

    above = daily > threshold
    if not above.any():
        return {"peak_magnitude": peak_magnitude, "pulse_duration": 0.0, "rise_rate": 0.0}

    peak_idx = int(np.argmax(daily))  # the global max is above threshold when any are
    lo = peak_idx
    while lo - 1 >= 0 and above[lo - 1]:
        lo -= 1
    hi = peak_idx
    while hi + 1 < len(daily) and above[hi + 1]:
        hi += 1
    pulse_duration = float(hi - lo + 1)

    # Rising limb: from one day before pulse onset up to the peak.
    onset = max(0, lo - 1)
    rising = daily[onset:peak_idx + 1]
    rise_rate = float(max(np.diff(rising).max(), 0.0) / ref_mean) if rising.size > 1 else 0.0
    return {
        "peak_magnitude": peak_magnitude,
        "pulse_duration": pulse_duration,
        "rise_rate": rise_rate,
    }


def compute_candidate_hazard_image(
    scenario_monthly: np.ndarray,
    scenario_daily: np.ndarray,
    reference_monthly: np.ndarray,
    reference_daily: np.ndarray,
    *,
    dry_timescale: int = 6,
    dist: str = "gamma",
    reference_start: str = "1945-10-01",
    dry_end_threshold: int = 3,
    dry_select: str = "controlling",
    flood_threshold_pct: float = 95.0,
) -> tuple[np.ndarray, list[str]]:
    """Compute the 6-candidate wet+dry event-descriptor hazard image.

    Dry axes: SSI-``dry_timescale`` (reservoir-drought timescale; SSI-6 default,
    McKee et al. 1993) on the monthly aggregate NYC inflow, controlling-event run
    theory. Wet axes: peaks-over-threshold on the daily aggregate NYC inflow with
    a ``flood_threshold_pct`` (Q5/95th-pct) threshold and mean-daily
    normalization, both fixed once on the historical reference.

    Args:
        scenario_monthly: ``(n, n_months)`` monthly aggregate NYC inflow.
        scenario_daily: ``(n, n_days)`` daily aggregate NYC inflow (same scenarios).
        reference_monthly: 1D historical monthly aggregate inflow (dry SSI fit).
        reference_daily: 1D historical daily aggregate inflow (flood threshold + mean).
        dry_timescale: Months accumulated for the drought SSI (SSI-6 default).
        dist: SSI fitting distribution.
        reference_start: October-aligned start for the dry SSI fit.
        dry_end_threshold: Drought-event recovery hysteresis (months).
        dry_select: Controlling-event selector (see :func:`critical_event_descriptors`).
        flood_threshold_pct: Percentile of the reference daily flow used as the
            POT high-flow threshold (95 = Q5).

    Returns:
        ``(H, list(CANDIDATE_EVENT_METRICS))`` with columns ordered dry then wet.
    """
    dry_calc = fit_reference_ssi(
        reference_monthly, timescale=dry_timescale, dist=dist, start_date=reference_start
    )
    ref_daily = np.asarray(reference_daily, dtype=float)
    threshold = float(np.percentile(ref_daily, flood_threshold_pct))
    ref_mean = float(ref_daily.mean())

    scenario_monthly = np.asarray(scenario_monthly, dtype=float)
    scenario_daily = np.asarray(scenario_daily, dtype=float)
    rows = []
    for m_row, d_row in zip(scenario_monthly, scenario_daily):
        dry_ssi = dry_calc.transform(flows_to_series(m_row, freq="MS"))
        d = critical_event_descriptors(dry_ssi, end_threshold=dry_end_threshold, select=dry_select)
        w = pot_flood_descriptors(d_row, threshold=threshold, ref_mean=ref_mean)
        rows.append([
            d["volume"], d["duration"], d["depth"], d["onset_rate"], d["recovery_rate"],
            w["peak_magnitude"], w["pulse_duration"], w["rise_rate"],
        ])
    return np.asarray(rows, dtype=float), list(CANDIDATE_EVENT_METRICS)
