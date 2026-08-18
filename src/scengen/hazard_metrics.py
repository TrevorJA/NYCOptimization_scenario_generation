"""Hazard metrics: the candidate event-descriptor image (methods 3.3).

Computes the per-scenario hazard vector ``h(r)`` that the hazard-filling design
selects on. The image is the **8-axis candidate event-descriptor set** of
:func:`compute_candidate_hazard_image`: five dry axes from SSI-6 run theory on
the controlling drought event of the monthly aggregate NYC inflow (magnitude,
duration, severity, onset rate, recovery rate) and three wet axes from
peaks-over-threshold on the daily aggregate NYC inflow (peak discharge, pulse
duration, rise rate). The two tails deliberately use different event models --
see the run-theory section below. The candidate set is then screened per pool to
a low-redundancy subset (Olden & Poff; :mod:`scengen.diagnostics` and
:func:`scengen.hazard_filling.select_from_candidate_image`), so the axes are
never hard-coded into the design.

The flow->series / SSI-calculator helpers are COPIED (not imported) from
MOEA-FIND (``src/metrics/objectives.py``) so this repo has no dependency on that
repo. SSI itself is provided by **SynHydro** (``synhydro.droughts.ssi``), an
allowed dependency, imported lazily so the module imports without it.

Design choices (methods 3.3):
  - The SSI gamma distribution is fit ONCE on the historical reference record
    (per calendar month), then applied to every scenario, so hazard
    coordinates are comparable across the pool and the historic reference. The
    POT threshold and mean-daily normalizer are likewise fixed on that record.
  - Series carry TRUE calendar dates (January starts: the reference record and
    every synthetic scenario begin on a January 1), so SSI's per-calendar-month
    fit applies to the correct months of each scenario. The scenario stamp and
    the reference start must share a start month; the image computation asserts
    this, because a one-sided drift degrades to silently clipped SSI rather
    than an error.
  - Both tails score the SAME effective window as the downstream objective
    metrics: the first six months of each scenario. The dry axes exclude it
    IMPLICITLY — SSI-6 needs six months of accumulation, so it is undefined
    there and no run-theory event can start in it — and the wet axes exclude it
    EXPLICITLY via ``wet_exclusion_days`` on the daily series. The monthly input
    is therefore never truncated: those months are the SSI accumulation input.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

#: Default pywrdrb nodes summed to form the aggregate NYC inflow series.
DEFAULT_NYC_INFLOW_NODES: tuple[str, ...] = ("cannonsville", "pepacton", "neversink")

#: Stamp for scenario rows entering the SSI transform. Only the month-of-year
#: matters, and it must equal the reference fit's start month (asserted in
#: :func:`compute_candidate_hazard_image`): scenarios and the reference record
#: both start on a true January 1.
_SCENARIO_STAMP_START = "2000-01-01"

#: True start of the historical reference record (calendar dates).
_REFERENCE_START = "1945-01-01"


def flows_to_series(
    monthly_flows: np.ndarray, start_date: str = _SCENARIO_STAMP_START, freq: str = "MS"
) -> pd.Series:
    """Wrap a 1D/2D monthly-flow array in a pd.Series with a DatetimeIndex.

    Copied from MOEA-FIND ``objectives.flows_to_series``. SynHydro's SSI requires
    a DatetimeIndex; ``start_date`` must share its month with the reference fit's
    start so the per-calendar-month SSI fit aligns with each scenario's months
    (both are true January starts).
    """
    arr = np.asarray(monthly_flows, dtype=float)
    if arr.ndim == 2:
        arr = arr.flatten()
    index = pd.date_range(start=start_date, periods=len(arr), freq=freq)
    return pd.Series(arr, index=index, name="flow_mgd")


def make_ssi_calculator(timescale: int = 6, dist: str = "gamma", fit_freq: str = "M"):
    """Create a configured (unfitted) SynHydro SSI calculator.

    Copied from MOEA-FIND ``objectives.make_ssi_calculator``. SynHydro is imported
    lazily so this module imports without it.
    """
    from synhydro.droughts.ssi import SSI  # lazy: allowed dependency
    return SSI(dist=dist, timescale=timescale, fit_freq=fit_freq)


def fit_reference_ssi(
    reference_monthly: np.ndarray,
    *,
    timescale: int = 6,
    dist: str = "gamma",
    start_date: str = _REFERENCE_START,
):
    """Fit an SSI calculator on the historical reference monthly series.

    Args:
        reference_monthly: 1D monthly aggregate-inflow array for the historical
            reference record (true calendar dates, January start).
        timescale: SSI accumulation period in months (SSI-6, the reservoir-drought
            timescale, is the default of the candidate image).
        dist: Fitting distribution (gamma default).
        start_date: True start date of the reference record.

    Returns:
        A fitted SynHydro SSI calculator to transform scenarios with.
    """
    calc = make_ssi_calculator(timescale=timescale, dist=dist)
    calc.fit(flows_to_series(reference_monthly, start_date=start_date))
    return calc


#: Content-keyed cache for the reference fits (see :func:`get_reference_fits`).
_REFERENCE_FIT_CACHE: dict = {}


def get_reference_fits(
    reference_monthly: np.ndarray,
    reference_daily: np.ndarray,
    *,
    timescale: int = 6,
    dist: str = "gamma",
    start_date: str = _REFERENCE_START,
    flood_threshold_pct: float = 95.0,
):
    """Fitted ``(dry_calc, threshold, ref_mean)`` for a reference record, cached.

    The reference fit is a deterministic pure function of the two reference
    arrays and the fit parameters, and every caller (generation hazard blocks,
    the E_test sub-window image, selection-time scoring) passes the same
    historical record on each call, so refitting per call is pure waste. Reuse
    is exactly result-preserving: ``SSI.fit`` is deterministic and
    ``SSI.transform`` does not mutate fitted state, so the cached object is the
    same object a fresh fit would produce. Keyed by a content hash of the
    arrays plus the fit parameters — never by object identity.

    Args:
        reference_monthly: 1D historical monthly aggregate inflow (SSI fit).
        reference_daily: 1D historical daily aggregate inflow (POT threshold + mean).
        timescale: SSI accumulation months.
        dist: SSI fitting distribution.
        start_date: True start date of the reference record.
        flood_threshold_pct: Percentile of the reference daily flow for the POT
            threshold.

    Returns:
        ``(dry_calc, threshold, ref_mean)`` — a fitted SSI calculator, the POT
        threshold (float), and the mean daily reference flow (float).
    """
    import hashlib

    ref_m = np.ascontiguousarray(np.asarray(reference_monthly, dtype=float))
    ref_d = np.ascontiguousarray(np.asarray(reference_daily, dtype=float))
    key = (
        hashlib.sha256(ref_m.tobytes()).hexdigest(),
        hashlib.sha256(ref_d.tobytes()).hexdigest(),
        int(timescale),
        str(dist),
        str(start_date),
        float(flood_threshold_pct),
    )
    hit = _REFERENCE_FIT_CACHE.get(key)
    if hit is None:
        dry_calc = fit_reference_ssi(
            ref_m, timescale=timescale, dist=dist, start_date=start_date
        )
        threshold = float(np.percentile(ref_d, flood_threshold_pct))
        ref_mean = float(ref_d.mean())
        hit = (dry_calc, threshold, ref_mean)
        _REFERENCE_FIT_CACHE[key] = hit
    return hit


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
    "drought_magnitude",       # |sum SSI over run| — cumulative-deficit facet
    "drought_duration",        # run length (months) — duration facet
    "drought_severity",        # |min SSI in run| — peak-deficit intensity
    "drought_onset_rate",      # |peak SSI| / months to trough — rate of change
    "drought_recovery_rate",   # |peak SSI| / recovery months — rate of change
)

#: Wet (flood / high-flow) candidate axes: critical peaks-over-threshold pulse
#: descriptors on the daily aggregate NYC inflow (Q5 / 95th-pct threshold).
WET_EVENT_METRICS: tuple[str, ...] = (
    "flood_peak_discharge",    # window max daily flow / reference mean — peak discharge
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
        Dict with non-negative ``duration``, ``magnitude`` (|cumulative SSI|), and
        ``severity`` (|peak SSI|). All zero when the scenario has no critical event.
    """
    from synhydro.droughts.ssi import get_drought_metrics  # lazy

    dm = get_drought_metrics(ssi_series, end_drought_threshold_months=end_threshold)
    if len(dm) == 0:
        return {"duration": 0.0, "magnitude": 0.0, "severity": 0.0,
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
    severity = float(abs(crit["severity"]))

    # Rate-of-change descriptors (orthogonal to magnitude/duration/severity).
    start = pd.Timestamp(crit["start"])
    peak = pd.Timestamp(crit["max_severity_date"])
    months_to_peak = max(1, (peak.year - start.year) * 12 + (peak.month - start.month))
    recovery_months = max(1.0, float(crit.get("recovery_period", 0.0)))
    return {
        "duration": float(crit["duration"]),
        "magnitude": float(abs(crit["magnitude"])),
        "severity": severity,
        "onset_rate": severity / months_to_peak,
        "recovery_rate": severity / recovery_months,
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
        Dict with ``peak_discharge`` (window max / ``ref_mean``; always defined,
        so no zero-inflation), ``pulse_duration`` (days above ``threshold`` in the
        critical pulse), and ``rise_rate`` (max 1-day rise on the critical rising
        limb / ``ref_mean``).
    """
    daily = np.asarray(daily, dtype=float)
    ref_mean = float(ref_mean) if ref_mean > 0 else 1.0
    peak_discharge = float(daily.max() / ref_mean)

    above = daily > threshold
    if not above.any():
        return {"peak_discharge": peak_discharge, "pulse_duration": 0.0, "rise_rate": 0.0}

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
        "peak_discharge": peak_discharge,
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
    reference_start: str = _REFERENCE_START,
    dry_end_threshold: int = 3,
    dry_select: str = "controlling",
    flood_threshold_pct: float = 95.0,
    wet_exclusion_days: int = 0,
    prefit_dry_calc=None,
    prefit_threshold: float | None = None,
    prefit_ref_mean: float | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Compute the 8-axis candidate wet+dry event-descriptor hazard image.

    Dry axes: SSI-``dry_timescale`` (reservoir-drought timescale; SSI-6 default,
    McKee et al. 1993) on the monthly aggregate NYC inflow, controlling-event run
    theory. Wet axes: peaks-over-threshold on the daily aggregate NYC inflow with
    a ``flood_threshold_pct`` (Q5/95th-pct) threshold and mean-daily
    normalization, both fixed once on the historical reference.

    Both tails describe the same effective window — the scenario minus its first
    six months. The dry axes exclude it implicitly (SSI-``dry_timescale``
    accumulation spin-up), the wet axes explicitly via ``wet_exclusion_days``.
    Only the SCENARIO daily window is truncated: the POT threshold and
    ``ref_mean`` stay fitted on the full historical reference, and the monthly
    input is never truncated because its leading months are the SSI accumulation
    input.

    Args:
        scenario_monthly: ``(n, n_months)`` monthly aggregate NYC inflow.
        scenario_daily: ``(n, n_days)`` daily aggregate NYC inflow (same scenarios).
        reference_monthly: 1D historical monthly aggregate inflow (dry SSI fit).
        reference_daily: 1D historical daily aggregate inflow (flood threshold + mean).
        dry_timescale: Months accumulated for the drought SSI (SSI-6 default).
        dist: SSI fitting distribution.
        reference_start: True start date of the reference record for the dry
            SSI fit. Its month must equal the scenario stamp's start month
            (asserted below): the fitted gammas are keyed by stamped calendar
            month, so a one-sided drift silently yields NaN/clipped SSI.
        dry_end_threshold: Drought-event recovery hysteresis (months).
        dry_select: Controlling-event selector (see :func:`critical_event_descriptors`).
        flood_threshold_pct: Percentile of the reference daily flow used as the
            POT high-flow threshold (95 = Q5).
        wet_exclusion_days: Leading days cut from each scenario's DAILY series
            before the POT descriptors, so the wet axes see the shared metric
            window. Callers compute it by date from the scenario's own
            DatetimeIndex (six months from a January start is 181 or 182 days).
        prefit_dry_calc: Optional already-fitted SSI calculator; when given
            (together with ``prefit_threshold`` and ``prefit_ref_mean``) the
            reference fit/percentile step is skipped entirely. Default None
            routes through the content-keyed :func:`get_reference_fits` cache.
        prefit_threshold: POT threshold matching ``prefit_dry_calc``.
        prefit_ref_mean: Mean daily reference flow matching ``prefit_dry_calc``.

    Returns:
        ``(H, list(CANDIDATE_EVENT_METRICS))`` with columns ordered dry then wet.

    Raises:
        ValueError: If ``wet_exclusion_days`` is negative or leaves no daily
            values in the scenario window.
    """
    # The fitted gammas are keyed by stamped calendar month on BOTH the fit and
    # the transform, so the scenario stamp and the reference start must share a
    # month; a mismatch produces silently NaN/clipped SSI, never an exception.
    if pd.Timestamp(reference_start).month != pd.Timestamp(_SCENARIO_STAMP_START).month:
        raise ValueError(
            f"reference_start={reference_start!r} and the scenario stamp "
            f"{_SCENARIO_STAMP_START!r} must share a start month; the per-calendar-"
            f"month SSI fit otherwise rotates against the scenarios."
        )
    if prefit_dry_calc is not None:
        if prefit_threshold is None or prefit_ref_mean is None:
            raise ValueError(
                "prefit_dry_calc requires prefit_threshold and prefit_ref_mean."
            )
        dry_calc = prefit_dry_calc
        threshold = float(prefit_threshold)
        ref_mean = float(prefit_ref_mean)
    else:
        dry_calc, threshold, ref_mean = get_reference_fits(
            reference_monthly,
            reference_daily,
            timescale=dry_timescale,
            dist=dist,
            start_date=reference_start,
            flood_threshold_pct=flood_threshold_pct,
        )

    scenario_monthly = np.asarray(scenario_monthly, dtype=float)
    scenario_daily = np.asarray(scenario_daily, dtype=float)
    cut = int(wet_exclusion_days)
    if cut < 0:
        raise ValueError(f"wet_exclusion_days must be >= 0, got {cut}")
    if cut >= scenario_daily.shape[-1]:
        raise ValueError(
            f"wet_exclusion_days={cut} leaves no daily values in a scenario window of "
            f"{scenario_daily.shape[-1]} days"
        )
    rows = []
    for m_row, d_row in zip(scenario_monthly, scenario_daily):
        dry_ssi = dry_calc.transform(flows_to_series(m_row, freq="MS"))
        d = critical_event_descriptors(dry_ssi, end_threshold=dry_end_threshold, select=dry_select)
        w = pot_flood_descriptors(d_row[cut:], threshold=threshold, ref_mean=ref_mean)
        rows.append([
            d["magnitude"], d["duration"], d["severity"], d["onset_rate"], d["recovery_rate"],
            w["peak_discharge"], w["pulse_duration"], w["rise_rate"],
        ])
    return np.asarray(rows, dtype=float), list(CANDIDATE_EVENT_METRICS)
