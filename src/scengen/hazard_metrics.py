"""Hazard metrics: the candidate event-descriptor image (methods 3.3).

Computes the per-scenario hazard vector ``h(r)`` that the hazard-filling design
selects on. The image is the **8-axis candidate event-descriptor set** of
:func:`compute_candidate_hazard_image`: five dry axes from SSI-6 run theory on
the controlling drought event of the monthly aggregate NYC inflow (magnitude,
duration, severity, development rate, termination rate) and three wet axes from
peaks-over-threshold on the daily aggregate NYC inflow (peak discharge, pulse
duration, rise rate). The two tails deliberately use different event models --
see the run-theory section below. The candidate set is then screened per pool to
a low-redundancy subset (Olden & Poff; :mod:`scengen.diagnostics` and
:func:`scengen.hazard_filling.select_from_candidate_image`), so the axes are
never hard-coded into the design.

Every image also carries a SUPPLEMENT (:data:`SUPPLEMENT_METRICS`): boundary
flags, event counts, low- and high-flow volumes and regime indices scored in the
same pass on the same window. The supplement never enters axis screening or
selection unless a caller names a column.

The flow->series / SSI-calculator helpers are COPIED (not imported) from
MOEA-FIND (``src/metrics/objectives.py``) so this repo has no dependency on that
repo. The SSI calculator is provided by **SynHydro** (``synhydro.droughts.ssi``),
an allowed dependency, imported lazily so the module imports without it; the
run-theory event model is implemented here (:func:`drought_events`).

Design choices (methods 3.3):
  - The SSI distribution is a two-parameter gamma (location fixed at zero,
    shape and scale by maximum likelihood; :class:`_ZeroLocationGamma`), fit
    ONCE on the historical reference record (per calendar month), then applied
    to every scenario, so hazard coordinates are comparable across the pool
    and the historic reference. A free location lets maximum likelihood
    degenerate on near-normal months, collapsing their SSI spread. The POT
    threshold and mean-daily normalizer are likewise fixed on that record.
  - Series carry TRUE calendar dates: the reference record starts on its own
    true date (a January 1), every synthetic scenario on the December-1
    realization epoch, and the scenario stamp carries that December start. The
    SSI fit and transform are both keyed by CALENDAR MONTH (spei's
    ``group_yearly_df`` remaps observations onto year-2000 dates and groups by
    month), so the reference and the scenarios may start in different months —
    what matters is that each stamp is truthful, or the per-month fit rotates
    against the content.
  - Both tails score the SAME effective window as the downstream objective
    metrics: the scenario minus its first six months (Dec – May, ending
    exactly on June 1, the FFMP operating-year boundary). The dry axes
    exclude it EXACTLY: the scored SSI series starts on the first month after
    the window (:func:`scored_dry_ssi`), so no run-theory event can start
    before June and none starting in June is missed. The wet axes exclude it
    EXPLICITLY via ``wet_exclusion_days`` on the daily series. The monthly
    input keeps its leading months: they are the SSI accumulation input.
    Callers cut the trailing partial year from both inputs, so the scored
    window is identical to the objectives' unit window. Every persisted
    hazard image records the dry cut (``_DRY_CUT_MONTHS``) and the scoring
    rules of the dry axes, the wet axes and the supplement
    (``_DRY_SCORING_RULE``, ``_WET_SCORING_RULE``,
    ``_SUPPLEMENT_SCORING_RULE``) as provenance, so an image scored on another
    window or under another rule is refused rather than mixed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

#: Default pywrdrb nodes summed to form the aggregate NYC inflow series.
DEFAULT_NYC_INFLOW_NODES: tuple[str, ...] = ("cannonsville", "pepacton", "neversink")

#: Stamp for scenario rows entering the SSI transform. Only the months-of-year
#: matter (the SSI fit is keyed by calendar month), and the stamp must carry
#: the scenarios' TRUE start month — a December 1, the realization epoch
#: (chosen so the 6-month exclusion ends on June 1, the FFMP operating-year
#: boundary). The reference record keeps its own true (January) start.
_SCENARIO_STAMP_START = "1999-12-01"

#: True start of the historical reference record (calendar dates).
_REFERENCE_START = "1945-01-01"

#: Leading months of every scenario absent from the scored dry series: the
#: SSI-6 accumulation window, which on the December-start scenarios ends on
#: June 1. Also the default SSI timescale, since the exclusion window IS the
#: accumulation window. Persisted with every hazard image (``dry_cut_months``)
#: and checked on load, so images scored on another dry window never mix.
_DRY_CUT_MONTHS = 6

#: Name of the dry-axis scoring rule: SSI from the two-parameter gamma, run
#: theory recording a qualifying event still open at the series end, and phase
#: rates scored on the part of the event inside the window. Persisted with
#: every hazard image (``dry_scoring_rule``) and checked on load, so images
#: scored under another rule never mix. Each rule below is renamed whenever
#: its scoring changes (``tests/test_scoring_rules.py`` pins the columns and
#: scored values under the name).
_DRY_SCORING_RULE = "gamma2-ssi+open-end-events+in-window-rates"

#: Name of the wet-axis scoring rule: peaks over the 95th percentile of the
#: reference daily flow, the critical pulse being the run above the threshold
#: that holds the window maximum, and the rise rate the largest one-day rise
#: on its rising limb. Persisted with every hazard image
#: (``wet_scoring_rule``) and checked on load.
_WET_SCORING_RULE = "q95-pot+critical-pulse+one-day-rise"

#: Name of the supplement scoring rule: the truncation flags and event totals
#: of the dry pass, the pulse totals of the wet pass, and the day-weighted
#: low-flow, high-flow and regime descriptors of the scored daily series.
#: Persisted with every hazard image (``supplement_scoring_rule``) and
#: checked on load.
_SUPPLEMENT_SCORING_RULE = "truncation-flags+event-totals+flow-regime"


class _ZeroLocationGamma:
    """Two-parameter gamma (location fixed at zero) in the form spei and SynHydro call.

    spei fits each calendar month with ``dist.fit(data, scale=..., method=...)``
    and reads back ``(shape, loc, scale)``; SynHydro's transform and spei's
    in-sample CDF evaluate ``dist.cdf(x, shape, loc, scale)`` with ``loc`` and
    ``scale`` positional or keyword. Both delegate to :data:`scipy.stats.gamma`,
    so the fit is maximum likelihood in shape and scale with the location
    held at zero.
    """

    def fit(self, data: np.ndarray | pd.Series, **kwargs) -> tuple:
        """Fit shape and scale with the location fixed at zero.

        Args:
            data: Values to fit (positive).
            **kwargs: Passed to :meth:`scipy.stats.gamma.fit` (spei's initial
                ``scale`` guess and fitting ``method``).

        Returns:
            ``(shape, 0.0, scale)``.
        """
        return stats.gamma.fit(data, floc=0.0, **kwargs)

    def cdf(self, x: np.ndarray | float, *args, **kwargs) -> np.ndarray | float:
        """Gamma CDF at ``x`` for the fitted ``(shape, loc, scale)``."""
        return stats.gamma.cdf(x, *args, **kwargs)


#: The SSI distribution of every calculator this module builds.
_SSI_DISTRIBUTION = _ZeroLocationGamma()


def flows_to_series(
    monthly_flows: np.ndarray, start_date: str = _SCENARIO_STAMP_START, freq: str = "MS"
) -> pd.Series:
    """Wrap a 1D/2D monthly-flow array in a pd.Series with a DatetimeIndex.

    Copied from MOEA-FIND ``objectives.flows_to_series``. SynHydro's SSI requires
    a DatetimeIndex; ``start_date`` must carry the array's TRUE start month so
    the per-calendar-month SSI fit applies to the correct months (the fit is
    month-keyed, so the reference and the scenarios may start in different
    months as long as each stamp is truthful).
    """
    arr = np.asarray(monthly_flows, dtype=float)
    if arr.ndim == 2:
        arr = arr.flatten()
    index = pd.date_range(start=start_date, periods=len(arr), freq=freq)
    return pd.Series(arr, index=index, name="flow_mgd")


def make_ssi_calculator(timescale: int = 6, fit_freq: str = "M"):
    """Create a configured (unfitted) SynHydro SSI calculator.

    Copied from MOEA-FIND ``objectives.make_ssi_calculator``, with the
    two-parameter gamma (:class:`_ZeroLocationGamma`) as the distribution.
    SynHydro is imported lazily so this module imports without it.
    """
    from synhydro.droughts.ssi import SSI  # lazy: allowed dependency
    return SSI(dist=_SSI_DISTRIBUTION, timescale=timescale, fit_freq=fit_freq)


def fit_reference_ssi(
    reference_monthly: np.ndarray,
    *,
    timescale: int = 6,
    start_date: str = _REFERENCE_START,
):
    """Fit an SSI calculator on the historical reference monthly series.

    Args:
        reference_monthly: 1D monthly aggregate-inflow array for the historical
            reference record (true calendar dates, January start).
        timescale: SSI accumulation period in months (SSI-6, the reservoir-drought
            timescale, is the default of the candidate image).
        start_date: True start date of the reference record.

    Returns:
        A fitted SynHydro SSI calculator (two-parameter gamma per calendar
        month) to transform scenarios with.
    """
    calc = make_ssi_calculator(timescale=timescale)
    calc.fit(flows_to_series(reference_monthly, start_date=start_date))
    return calc


#: Content-keyed cache for the reference fits (see :func:`get_reference_fits`).
_REFERENCE_FIT_CACHE: dict = {}


def get_reference_fits(
    reference_monthly: np.ndarray,
    reference_daily: np.ndarray,
    *,
    timescale: int = 6,
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
        str(start_date),
        float(flood_threshold_pct),
    )
    hit = _REFERENCE_FIT_CACHE.get(key)
    if hit is None:
        dry_calc = fit_reference_ssi(ref_m, timescale=timescale, start_date=start_date)
        threshold = float(np.percentile(ref_d, flood_threshold_pct))
        ref_mean = float(ref_d.mean())
        hit = (dry_calc, threshold, ref_mean)
        _REFERENCE_FIT_CACHE[key] = hit
    return hit


def scored_dry_ssi(
    dry_calc, monthly_row: np.ndarray, *, dry_timescale: int = _DRY_CUT_MONTHS
) -> tuple[pd.Series, float | None]:
    """SSI series of one scenario on the dry-axis scoring window, and the month before it.

    Transforms the December-stamped monthly series with the fitted calculator
    and returns it from exactly ``dry_timescale`` months after the scenario
    start (June 1 on the December-start scenarios), the first month after the
    exclusion window. The transform returns only the months on which SSI is
    defined, so the cut applied here is whatever it left of the exclusion
    window; the result is the same whether the transform drops the undefined
    leading months or returns them as NaN. The SSI of the last excluded month
    (May on the December-start scenarios) comes back with the series: run
    theory needs it to tell whether an event under way in the first scored
    month began before the window.

    Args:
        dry_calc: Fitted SynHydro SSI calculator (:func:`fit_reference_ssi`).
        monthly_row: 1D monthly aggregate NYC inflow of one scenario, leading
            months included (they are the accumulation input).
        dry_timescale: SSI accumulation months, which is also the length of
            the exclusion window.

    Returns:
        ``(ssi, ssi_pre)``: the SSI series stamped from the first scored month,
        one value per month of the scoring window, and the SSI of the month
        immediately before it (None when the transform leaves it undefined).

    Raises:
        ValueError: If the transform dropped more leading months than the
            exclusion window holds, or the scored series does not start on
            the first month after that window.
    """
    monthly_row = np.asarray(monthly_row, dtype=float)
    full = dry_calc.transform(flows_to_series(monthly_row, freq="MS"))
    already_dropped = len(monthly_row) - len(full)
    remaining = int(dry_timescale) - already_dropped
    if remaining < 0:
        raise ValueError(
            f"the SSI transform dropped {already_dropped} leading months, more than "
            f"the {int(dry_timescale)}-month exclusion window."
        )
    ssi = full.iloc[remaining:]
    expected = pd.Timestamp(_SCENARIO_STAMP_START) + pd.DateOffset(months=int(dry_timescale))
    if len(ssi) == 0 or ssi.index[0] != expected:
        raise ValueError(
            f"the scored SSI series must start on {expected.date()}, the first month "
            f"after the exclusion window; got "
            f"{ssi.index[0].date() if len(ssi) else 'an empty series'}."
        )
    ssi_pre = float(full.iloc[remaining - 1]) if remaining > 0 else np.nan
    return ssi, (ssi_pre if np.isfinite(ssi_pre) else None)


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
#: theory: deficit ~ duration x intensity); the phase rates (the event divided
#: at its minimum; the termination rate adapted from Parry et al. 2016, the
#: development rate its onset-side counterpart and the rate-of-intensification
#: facet of the flash-drought literature, Otkin et al. 2018) are the orthogonal
#: facet that yields a genuine SECOND drought axis after screening.
DRY_EVENT_METRICS: tuple[str, ...] = (
    "drought_magnitude",          # |sum SSI over run| — cumulative-deficit facet
    "drought_duration",           # run length (months) — duration facet
    "drought_severity",           # |min SSI in run| — peak-deficit intensity
    "drought_development_rate",   # SSI decline per month, onset to minimum
    "drought_termination_rate",   # SSI recovery per month, minimum to end
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

#: Supplementary descriptors carried with every hazard image as a second array
#: (:func:`compute_candidate_hazard_image` with ``return_supplement=True``).
#: Scored on the same window as the axes; flows are over the reference mean
#: daily flow, so values are dimensionless; flags are 0.0 / 1.0. They never
#: enter axis screening or selection unless a caller names a column.
SUPPLEMENT_METRICS: tuple[str, ...] = (
    "drought_onset_truncated",        # controlling event under way before the window
    "drought_termination_truncated",  # controlling event still below zero at the window end
    "drought_event_count",            # qualifying run-theory events
    "drought_total_deficit",          # summed magnitude of all qualifying events
    "lowflow_min_12month",            # minimum 12-month mean flow
    "lowflow_min_24month",            # minimum 24-month mean flow
    "lowflow_min_year",               # lowest year (12-month block) mean flow
    "lowflow_min_7day",               # minimum 7-day mean flow
    "flood_pulse_count",              # runs above the POT threshold (no declustering)
    "flood_days_above",               # days above the POT threshold
    "flood_max_3day",                 # maximum 3-day mean flow
    "flood_pulse_volume",             # critical-pulse volume above threshold (ref-mean days)
    "highflow_max_year",              # highest year mean flow
    "annual_cv",                      # coefficient of variation of the year means
    "flashiness",                     # Richards-Baker index of the daily flow
)


@dataclass(frozen=True)
class DroughtEvent:
    """One run-theory drought event; positions index the scored SSI series.

    Attributes:
        start: Position of the first month below zero.
        end: Position of the last month below zero.
        peak: Position of the first occurrence of the event minimum.
        duration: Months below zero (non-negative interludes excluded).
        magnitude: Absolute sum of the below-zero values.
        severity: Absolute value of the minimum.
    """

    start: int
    end: int
    peak: int
    duration: int
    magnitude: float
    severity: float


def drought_events(
    ssi_series: np.ndarray | pd.Series, *, end_threshold: int = 3
) -> list[DroughtEvent]:
    """Run-theory drought events of a scored SSI series (Yevjevich 1967).

    Month by month: a month with SSI < 0 joins the current run and resets the
    recovery count, and the run qualifies once any of its months is <= -1. A
    non-negative month advances the recovery count of a qualifying run and
    closes the event when the count reaches ``end_threshold``; it clears a run
    that has not qualified. Non-negative months inside a qualifying run are not
    part of its duration or magnitude. A qualifying run still open when the
    series ends is recorded.

    Args:
        ssi_series: 1D scored SSI series (array or Series).
        end_threshold: Consecutive non-negative months that close an event.

    Returns:
        The qualifying events in chronological order.

    Raises:
        ValueError: If the series contains NaN.
    """
    z = np.asarray(ssi_series, dtype=float)
    if np.isnan(z).any():
        raise ValueError("the SSI series contains NaN; run theory needs every month.")
    events: list[DroughtEvent] = []
    run: list[int] = []
    qualifying = False
    recovery = 0
    for i, value in enumerate(z):
        if value < 0.0:
            run.append(i)
            recovery = 0
            qualifying = qualifying or value <= -1.0
        elif qualifying:
            recovery += 1
            if recovery >= end_threshold:
                events.append(_drought_event(z, run))
                run, qualifying, recovery = [], False, 0
        else:
            run = []
    if qualifying:
        events.append(_drought_event(z, run))
    return events


def _drought_event(z: np.ndarray, run: list[int]) -> DroughtEvent:
    """Descriptors of one qualifying run given the positions of its below-zero months."""
    months = np.asarray(run)
    values = z[months]
    k = int(np.argmin(values))  # first occurrence of the minimum
    return DroughtEvent(
        start=int(months[0]),
        end=int(months[-1]),
        peak=int(months[k]),
        duration=int(months.size),
        magnitude=float(abs(values.sum())),
        severity=float(abs(values[k])),
    )


def select_drought_event(
    events: list[DroughtEvent], select: str = "controlling"
) -> DroughtEvent | None:
    """The window's critical drought event.

    Args:
        events: Events from :func:`drought_events`.
        select: ``"controlling"`` (largest magnitude; default), ``"deepest"``
            (largest severity), or ``"first"`` (earliest event).

    Returns:
        The selected event (the earliest on a tie), or None without events.

    Raises:
        ValueError: If ``select`` is unknown.
    """
    if select not in ("controlling", "deepest", "first"):
        raise ValueError(f"unknown select={select!r}")
    if not events:
        return None
    if select == "first":
        return events[0]
    attribute = "magnitude" if select == "controlling" else "severity"
    return max(events, key=lambda e: getattr(e, attribute))


def critical_event_descriptors(
    ssi_series: np.ndarray | pd.Series,
    *,
    ssi_pre: float | None = None,
    end_threshold: int = 3,
    select: str = "controlling",
) -> dict:
    """Run-theory descriptors of the window's controlling drought event.

    Events come from :func:`drought_events`. The default selector is
    ``"controlling"``, the event with the largest cumulative deficit (the
    operationally binding event under fixed initial storage), which is more
    stable than the deepest-severity argmax.

    The event is divided at its minimum into a development and a termination
    phase. With ``start`` the first month below zero, ``peak`` the minimum and
    ``end`` the last month below zero (positions in the ``n``-month scored
    series ``z``), an observed phase's rate is the severity over its elapsed
    months, ``peak - start + 1`` and ``end - peak + 1`` (the crossing month
    counts; non-negative interludes inside the event count as elapsed time).
    The onset is truncated when the event starts at position 0 and
    ``ssi_pre`` is negative or unavailable; the termination is truncated when
    ``end`` is position ``n - 1``. A truncated development rate is
    ``(z[start] - z[peak]) / (peak - start)`` and a truncated termination rate
    ``(z[end] - z[peak]) / (end - peak)``, zero when the minimum lies on that
    window edge. Magnitude, duration and severity of a truncated event are
    scored on its in-window months.

    Every descriptor describes the part of the event inside the scored window,
    which is the part the scored simulation experiences. For a truncated phase
    the zero crossing is unobserved, so the rate is the index change observed
    within the window over the elapsed months. The termination rate is adapted
    from Parry et al. (2016); the development rate applies the same
    construction to the onset side.

    Args:
        ssi_series: Scored SSI series (array or Series).
        ssi_pre: SSI of the month immediately before the first scored month
            (:func:`scored_dry_ssi`); None when unavailable, in which case an
            event starting at position 0 counts as onset-truncated.
        end_threshold: Consecutive non-negative months that close an event.
        select: Critical-event selector (see :func:`select_drought_event`).

    Returns:
        Dict with the selected event's non-negative ``magnitude`` (|cumulative
        SSI|), ``duration`` (months below zero), ``severity`` (|minimum SSI|),
        ``development_rate`` and ``termination_rate`` (SSI units per month),
        its boolean ``onset_truncated`` and ``termination_truncated``, and the
        window's ``event_count`` and ``total_deficit`` (summed magnitude of
        every event). Zero (flags False) when the window has no event.
    """
    z = np.asarray(ssi_series, dtype=float)
    events = drought_events(z, end_threshold=end_threshold)
    event = select_drought_event(events, select)
    window = {
        "event_count": float(len(events)),
        "total_deficit": float(sum(e.magnitude for e in events)),
    }
    if event is None:
        return {
            "magnitude": 0.0, "duration": 0.0, "severity": 0.0,
            "development_rate": 0.0, "termination_rate": 0.0,
            "onset_truncated": False, "termination_truncated": False, **window,
        }

    pre_negative = ssi_pre is None or np.isnan(ssi_pre) or ssi_pre < 0.0
    onset_truncated = event.start == 0 and pre_negative
    termination_truncated = event.end == len(z) - 1
    if not onset_truncated:
        development_rate = event.severity / (event.peak - event.start + 1)
    elif event.peak > event.start:
        development_rate = (z[event.start] - z[event.peak]) / (event.peak - event.start)
    else:
        development_rate = 0.0
    if not termination_truncated:
        termination_rate = event.severity / (event.end - event.peak + 1)
    elif event.end > event.peak:
        termination_rate = (z[event.end] - z[event.peak]) / (event.end - event.peak)
    else:
        termination_rate = 0.0
    return {
        "magnitude": event.magnitude,
        "duration": float(event.duration),
        "severity": event.severity,
        "development_rate": float(development_rate),
        "termination_rate": float(termination_rate),
        "onset_truncated": bool(onset_truncated),
        "termination_truncated": bool(termination_truncated),
        **window,
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
        limb / ``ref_mean``); and for the supplement ``pulse_count`` (runs above
        ``threshold``, no declustering), ``days_above`` (days above
        ``threshold``) and ``pulse_volume`` (critical-pulse flow above
        ``threshold`` summed over its days / ``ref_mean``).
    """
    daily = np.asarray(daily, dtype=float)
    ref_mean = float(ref_mean) if ref_mean > 0 else 1.0
    peak_discharge = float(daily.max() / ref_mean)

    above = daily > threshold
    if not above.any():
        return {
            "peak_discharge": peak_discharge, "pulse_duration": 0.0, "rise_rate": 0.0,
            "pulse_count": 0.0, "days_above": 0.0, "pulse_volume": 0.0,
        }

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
        "pulse_count": float(above[0] + np.count_nonzero(above[1:] & ~above[:-1])),
        "days_above": float(np.count_nonzero(above)),
        "pulse_volume": float((daily[lo:hi + 1] - threshold).sum() / ref_mean),
    }


def flow_regime_descriptors(
    daily: np.ndarray, month_days: np.ndarray, *, ref_mean: float
) -> dict:
    """Low-flow, high-flow and regime descriptors of one scored window (supplement).

    Month and year means are day-weighted: the daily flow summed over
    consecutive calendar months divided by their day count. Years are the
    complete 12-month blocks counted from the window start (June to May on the
    production window); trailing months short of a block enter only the
    rolling month means.

    Args:
        daily: 1D daily flow over the scored window.
        month_days: Day count of each consecutive calendar month of the window
            (at least 24 months, summing to ``len(daily)``).
        ref_mean: Reference mean daily flow normalizing the flows.

    Returns:
        Dict keyed by supplement column: ``lowflow_min_12month`` and
        ``lowflow_min_24month`` (minimum rolling 12- and 24-month mean),
        ``lowflow_min_year`` and ``highflow_max_year`` (lowest and highest year
        mean), ``lowflow_min_7day`` (minimum 7-day mean) and ``flood_max_3day``
        (maximum 3-day mean), all over ``ref_mean``; ``annual_cv`` (coefficient
        of variation of the year means, ``ddof=1``) and ``flashiness``
        (Richards-Baker index, Baker et al. 2004: summed absolute day-to-day
        change over summed flow).

    Raises:
        ValueError: If ``month_days`` spans fewer than 24 months or does not
            sum to ``len(daily)``.
    """
    x = np.asarray(daily, dtype=float)
    month_days = np.asarray(month_days, dtype=int)
    if month_days.size < 24 or month_days.sum() != x.size:
        raise ValueError(
            f"month_days must hold at least 24 months summing to the {x.size} daily "
            f"values; got {month_days.size} months summing to {month_days.sum()}."
        )
    ref_mean = float(ref_mean) if ref_mean > 0 else 1.0
    csum = np.concatenate([[0.0], np.cumsum(x)])
    edges = np.concatenate([[0], np.cumsum(month_days)])  # day offset of each month start

    def month_means(k: int) -> np.ndarray:
        """Day-weighted mean flow of every run of ``k`` consecutive months."""
        return (csum[edges[k:]] - csum[edges[:-k]]) / (edges[k:] - edges[:-k])

    def day_means(k: int) -> np.ndarray:
        """Mean flow of every run of ``k`` consecutive days."""
        return (csum[k:] - csum[:-k]) / k

    years = month_means(12)[::12]
    return {
        "lowflow_min_12month": float(month_means(12).min() / ref_mean),
        "lowflow_min_24month": float(month_means(24).min() / ref_mean),
        "lowflow_min_year": float(years.min() / ref_mean),
        "lowflow_min_7day": float(day_means(7).min() / ref_mean),
        "flood_max_3day": float(day_means(3).max() / ref_mean),
        "highflow_max_year": float(years.max() / ref_mean),
        "annual_cv": float(years.std(ddof=1) / years.mean()),
        "flashiness": float(np.abs(np.diff(x)).sum() / x.sum()),
    }


def _scored_month_days(
    scenario_start: str | pd.Timestamp,
    *,
    n_months: int,
    n_days: int,
    dry_timescale: int,
    wet_exclusion_days: int,
) -> np.ndarray:
    """Day count of each scored month, checking the inputs against the true calendar.

    Args:
        scenario_start: True date of the first value of every scenario row.
        n_months: Months in each scenario's monthly input (excluded months included).
        n_days: Days in each scenario's daily input (excluded days included).
        dry_timescale: Leading months excluded from the scored window.
        wet_exclusion_days: Leading days cut from the daily input.

    Returns:
        Integer day counts of the ``n_months - dry_timescale`` scored months.

    Raises:
        ValueError: If ``scenario_start`` is not the first day of the scenario
            stamp's month, or the daily input minus its cut does not span
            exactly the scored months.
    """
    start = pd.Timestamp(scenario_start)
    stamp_month = pd.Timestamp(_SCENARIO_STAMP_START).month
    if not start.is_month_start or start.month != stamp_month:
        raise ValueError(
            f"scenario_start={start.date()} must be the first day of calendar month "
            f"{stamp_month}, the month the scenario stamp carries."
        )
    edges = pd.date_range(
        start + pd.DateOffset(months=int(dry_timescale)),
        periods=n_months - int(dry_timescale) + 1, freq="MS",
    )
    if (edges[0] - start).days != wet_exclusion_days or (
        (edges[-1] - edges[0]).days != n_days - wet_exclusion_days
    ):
        raise ValueError(
            f"the daily input ({n_days} days, {wet_exclusion_days} excluded) does not span "
            f"exactly the {n_months - int(dry_timescale)} scored months from "
            f"{edges[0].date()}."
        )
    return np.diff(edges.to_numpy()).astype("timedelta64[D]").astype(int)


def compute_candidate_hazard_image(
    scenario_monthly: np.ndarray,
    scenario_daily: np.ndarray,
    reference_monthly: np.ndarray,
    reference_daily: np.ndarray,
    *,
    dry_timescale: int = _DRY_CUT_MONTHS,
    reference_start: str = _REFERENCE_START,
    dry_end_threshold: int = 3,
    dry_select: str = "controlling",
    flood_threshold_pct: float = 95.0,
    wet_exclusion_days: int = 0,
    prefit_dry_calc=None,
    prefit_threshold: float | None = None,
    prefit_ref_mean: float | None = None,
    return_supplement: bool = False,
    scenario_start: str | pd.Timestamp | None = None,
) -> tuple[np.ndarray, list[str]] | tuple[np.ndarray, list[str], np.ndarray, list[str]]:
    """Compute the 8-axis candidate wet+dry event-descriptor hazard image.

    Dry axes: SSI-``dry_timescale`` (reservoir-drought timescale; SSI-6 default,
    McKee et al. 1993) on the monthly aggregate NYC inflow, controlling-event run
    theory. Wet axes: peaks-over-threshold on the daily aggregate NYC inflow with
    a ``flood_threshold_pct`` (Q5/95th-pct) threshold and mean-daily
    normalization, both fixed once on the historical reference.

    Both tails describe the same effective window — the scenario minus its
    first ``dry_timescale`` months. The dry axes exclude it exactly: run
    theory sees the SSI series from the first month after the window
    (:func:`scored_dry_ssi`). The wet axes exclude it explicitly via
    ``wet_exclusion_days``. Only the SCENARIO windows are truncated: the POT
    threshold and ``ref_mean`` stay fitted on the full historical reference,
    and the scenario monthly input keeps its leading months as the SSI
    accumulation input (callers cut any trailing partial year from both
    scenario inputs before this function).

    The supplement (:data:`SUPPLEMENT_METRICS`) comes from the same pass over
    each scenario and the same scored window: the dry-axis flags and event
    totals from run theory, the flood-pulse totals from the POT pass, and the
    low-flow, high-flow and regime descriptors from the scored daily series
    (:func:`flow_regime_descriptors`), whose day-weighted month and year means
    follow the true calendar given by ``scenario_start``.

    Args:
        scenario_monthly: ``(n, n_months)`` monthly aggregate NYC inflow.
        scenario_daily: ``(n, n_days)`` daily aggregate NYC inflow (same scenarios).
        reference_monthly: 1D historical monthly aggregate inflow (dry SSI fit).
        reference_daily: 1D historical daily aggregate inflow (flood threshold + mean).
        dry_timescale: Months accumulated for the drought SSI (SSI-6 default).
        reference_start: True start date of the reference record for the dry
            SSI fit. The fitted gammas are keyed by CALENDAR MONTH (spei maps
            observations onto year-2000 dates and groups by month), so the
            reference and the scenario stamp may start in different months —
            each stamp just has to be truthful for its own content.
        dry_end_threshold: Drought-event recovery hysteresis (months).
        dry_select: Critical-event selector (see :func:`select_drought_event`).
        flood_threshold_pct: Percentile of the reference daily flow used as the
            POT high-flow threshold (95 = Q5).
        wet_exclusion_days: Leading days cut from each scenario's DAILY series
            before the POT descriptors, so the wet axes see the shared metric
            window. Callers compute it by date from the scenario's own
            DatetimeIndex (six months from a December start is 182 or 183
            days).
        prefit_dry_calc: Optional already-fitted SSI calculator; when given
            (together with ``prefit_threshold`` and ``prefit_ref_mean``) the
            reference fit/percentile step is skipped entirely. Default None
            routes through the content-keyed :func:`get_reference_fits` cache.
        prefit_threshold: POT threshold matching ``prefit_dry_calc``.
        prefit_ref_mean: Mean daily reference flow matching ``prefit_dry_calc``.
        return_supplement: Also return the supplement array and its names.
        scenario_start: True date of the first value of every scenario row (all
            rows share one calendar; a December 1, the realization epoch or a
            window anchor). Required with ``return_supplement``, which then
            also requires ``wet_exclusion_days`` to end on the first scored
            month and the daily input to span exactly the scored months.

    Returns:
        ``(H, list(CANDIDATE_EVENT_METRICS))`` with columns ordered dry then wet;
        with ``return_supplement``, ``(H, list(CANDIDATE_EVENT_METRICS), S,
        list(SUPPLEMENT_METRICS))`` with ``S`` of shape ``(n, 15)``.

    Raises:
        ValueError: If ``wet_exclusion_days`` is negative or leaves no daily
            values in the scenario window, or ``return_supplement`` is set
            without a ``scenario_start`` consistent with the inputs.
    """
    # The fitted gammas are keyed by CALENDAR MONTH on both the fit and the
    # transform (spei's group_yearly_df maps every observation onto its
    # year-2000 date and groups by month), so the reference and the scenario
    # stamp need NOT share a start month. Each stamp must simply be truthful
    # for its own content — the reference's true start and the scenarios'
    # December epoch — which the callers' generation-time anchor assertions
    # guarantee.
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
    if return_supplement:
        if scenario_start is None:
            raise ValueError("return_supplement requires scenario_start.")
        month_days = _scored_month_days(
            scenario_start,
            n_months=scenario_monthly.shape[-1],
            n_days=scenario_daily.shape[-1],
            dry_timescale=dry_timescale,
            wet_exclusion_days=cut,
        )
    rows, supplement_rows = [], []
    for m_row, d_row in zip(scenario_monthly, scenario_daily):
        # Run theory scores the SSI series from the first month after the
        # exclusion window (June 1 on the December-start scenarios).
        dry_ssi, ssi_pre = scored_dry_ssi(dry_calc, m_row, dry_timescale=dry_timescale)
        d = critical_event_descriptors(
            dry_ssi, ssi_pre=ssi_pre, end_threshold=dry_end_threshold, select=dry_select
        )
        w = pot_flood_descriptors(d_row[cut:], threshold=threshold, ref_mean=ref_mean)
        values = {f"drought_{k}": v for k, v in d.items()}
        values |= {f"flood_{k}": v for k, v in w.items()}
        rows.append([values[m] for m in CANDIDATE_EVENT_METRICS])
        if return_supplement:
            values |= flow_regime_descriptors(d_row[cut:], month_days, ref_mean=ref_mean)
            supplement_rows.append([float(values[m]) for m in SUPPLEMENT_METRICS])
    H = np.asarray(rows, dtype=float)
    if not return_supplement:
        return H, list(CANDIDATE_EVENT_METRICS)
    S = np.asarray(supplement_rows, dtype=float)
    return H, list(CANDIDATE_EVENT_METRICS), S, list(SUPPLEMENT_METRICS)
