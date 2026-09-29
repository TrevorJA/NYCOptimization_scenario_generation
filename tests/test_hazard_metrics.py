"""Tests for the candidate event-descriptor hazard image. Skipped without SynHydro."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("synhydro", reason="hazard_metrics needs SynHydro's SSI")

from scengen import hazard_metrics as hm  # noqa: E402


def _synthetic_monthly(n_years, *, seed, scale=100.0, dip=None):
    """Gamma-distributed monthly flows (January-start years), optional drought dip."""
    rng = np.random.default_rng(seed)
    flows = rng.gamma(shape=2.0, scale=scale, size=n_years * 12)
    if dip is not None:
        lo, hi = dip
        flows[lo:hi] *= 0.2  # impose a multi-month low-flow event
    return flows


def _seasonal_monthly(n_years, *, seed, amplitude=0.8):
    """Monthly flows with a strong seasonal cycle peaking in April (spring melt)."""
    rng = np.random.default_rng(seed)
    m = np.arange(n_years * 12) % 12  # 0 = January
    cycle = 1.0 + amplitude * np.cos(2.0 * np.pi * (m - 3) / 12.0)  # peak at m=3 (April)
    return rng.gamma(shape=8.0, scale=100.0 / 8.0, size=n_years * 12) * cycle


def _december_monthly(n_years, *, deficit=None, surplus=None, base=200.0):
    """Constant monthly flows at the reference mean on the December scenario
    stamp with the trailing partial year cut (``12 n_years - 6`` months);
    ``deficit`` / ``surplus`` are index ranges scaled to 2 % / ten-fold of
    ``base``. Index 0 is December of year 0 and index 6 is June of year 1."""
    flows = np.full(12 * n_years - 6, base)
    if deficit is not None:
        flows[slice(*deficit)] *= 0.02
    if surplus is not None:
        flows[slice(*surplus)] *= 10.0
    return flows


def _synthetic_daily(n_years, *, seed, scale=100.0, spike=None):
    """Gamma-distributed daily flows, optional high-flow pulse."""
    rng = np.random.default_rng(seed + 1000)
    flows = rng.gamma(shape=2.0, scale=scale, size=n_years * 365)
    if spike is not None:
        lo, hi = spike
        flows[lo:hi] *= 10.0
    return flows


def test_critical_event_descriptors_are_nonnegative_and_complete():
    """Run-theory descriptors of the controlling drought event."""
    ref = _synthetic_monthly(78, seed=0)
    calc = hm.fit_reference_ssi(ref)
    ssi = calc.transform(hm.flows_to_series(_synthetic_monthly(5, seed=1, dip=(24, 36))))
    out = hm.critical_event_descriptors(ssi)
    assert set(out) == {
        "duration", "magnitude", "severity", "development_rate", "termination_rate",
        "onset_truncated", "termination_truncated", "event_count", "total_deficit",
    }
    assert all(np.isfinite(v) and v >= 0.0 for v in out.values())


def test_drought_scenario_has_larger_magnitude():
    """An imposed multi-month low-flow dip must deepen the controlling event."""
    ref = _synthetic_monthly(78, seed=0)
    calc = hm.fit_reference_ssi(ref)

    def magnitude(flows):
        return hm.critical_event_descriptors(
            calc.transform(hm.flows_to_series(flows))
        )["magnitude"]

    normal = magnitude(_synthetic_monthly(5, seed=2))
    drought = magnitude(_synthetic_monthly(5, seed=2, dip=(24, 36)))
    assert drought > normal


def test_scored_dry_series_starts_in_june_after_the_exclusion_window():
    """The dry axes score exactly the scenario minus its first six months: one
    value per month from June of year 1 (the first month after the exclusion
    window on the December stamp) to the last input month, and run theory on
    that series is what the candidate image reports."""
    ref_m, ref_d = _synthetic_monthly(78, seed=0), _synthetic_daily(78, seed=0)
    calc = hm.fit_reference_ssi(ref_m)
    n_months = 12 * 5 - 6
    scen = _synthetic_monthly(5, seed=1)[:n_months]

    ssi, ssi_pre = hm.scored_dry_ssi(calc, scen)
    assert len(ssi) == n_months - 6
    assert (ssi.index[0].year, ssi.index[0].month, ssi.index[0].day) == (
        pd.Timestamp(hm._SCENARIO_STAMP_START).year + 1, 6, 1)
    assert ssi.index[-1].month == 5
    assert not ssi.isna().any()
    # The month before the window is the last excluded month (May of year 1).
    full = calc.transform(hm.flows_to_series(scen))
    assert ssi_pre == full.loc[ssi.index[0] - pd.DateOffset(months=1)]

    H, names = hm.compute_candidate_hazard_image(
        scen[None, :], _synthetic_daily(5, seed=1)[None, :], ref_m, ref_d,
    )
    d = hm.critical_event_descriptors(ssi, ssi_pre=ssi_pre)
    expected = [d[m.removeprefix("drought_")] for m in hm.DRY_EVENT_METRICS]
    np.testing.assert_allclose(H[0, :len(hm.DRY_EVENT_METRICS)], expected)


def test_deficit_in_june_to_october_of_year_one_is_scored():
    """A deep deficit confined to June - October of year 1, the first five
    scored months, reaches the dry axes as a five-month controlling event.
    The ten-fold surplus from November returns the six-month accumulation to
    normal at once, so the event holds nothing outside those months."""
    ref_m, ref_d = _synthetic_monthly(78, seed=0), _synthetic_daily(78, seed=0)
    scen_m = np.vstack([_december_monthly(5, deficit=(6, 11), surplus=(11, 17))])
    scen_d = np.vstack([_synthetic_daily(5, seed=1)])

    H, names = hm.compute_candidate_hazard_image(scen_m, scen_d, ref_m, ref_d)
    assert H[0, names.index("drought_magnitude")] > 0.0
    assert H[0, names.index("drought_duration")] == 5.0


def test_deficit_inside_the_exclusion_window_is_not_scored():
    """A deficit confined to December - May of the exclusion window leaves the
    dry axes at zero; the ten-fold surplus over the following six months
    returns the accumulation to normal on the first scored month."""
    ref_m, ref_d = _synthetic_monthly(78, seed=0), _synthetic_daily(78, seed=0)
    scen_m = np.vstack([_december_monthly(5, deficit=(0, 6), surplus=(6, 12))])
    scen_d = np.vstack([_synthetic_daily(5, seed=2)])

    H, _ = hm.compute_candidate_hazard_image(scen_m, scen_d, ref_m, ref_d)
    assert np.all(H[0, :len(hm.DRY_EVENT_METRICS)] == 0.0)


def test_pot_flood_descriptors_respond_to_a_pulse():
    """POT descriptors of the critical high-flow pulse (daily series, no SSI)."""
    ref_d = _synthetic_daily(78, seed=0)
    threshold = float(np.percentile(ref_d, 95.0))
    ref_mean = float(ref_d.mean())

    plain = hm.pot_flood_descriptors(
        _synthetic_daily(5, seed=1), threshold=threshold, ref_mean=ref_mean
    )
    spiked = hm.pot_flood_descriptors(
        _synthetic_daily(5, seed=1, spike=(400, 410)),
        threshold=threshold, ref_mean=ref_mean,
    )
    assert set(plain) == {
        "peak_discharge", "pulse_duration", "rise_rate",
        "pulse_count", "days_above", "pulse_volume",
    }
    assert spiked["peak_discharge"] > plain["peak_discharge"]
    assert spiked["pulse_duration"] >= plain["pulse_duration"]


def test_candidate_hazard_image_shape_and_axes():
    """The live pipeline entry: an 8-axis (5 dry + 3 wet) candidate image."""
    ref_m, ref_d = _synthetic_monthly(78, seed=0), _synthetic_daily(78, seed=0)
    scen_m = np.vstack([_synthetic_monthly(5, seed=s) for s in range(8)])
    scen_d = np.vstack([_synthetic_daily(5, seed=s) for s in range(8)])

    H, names = hm.compute_candidate_hazard_image(scen_m, scen_d, ref_m, ref_d)

    assert names == list(hm.CANDIDATE_EVENT_METRICS)
    assert names == list(hm.DRY_EVENT_METRICS) + list(hm.WET_EVENT_METRICS)
    assert H.shape == (8, 8)
    assert np.isfinite(H).all()
    assert (H >= 0.0).all()


def test_wet_exclusion_hides_a_flood_inside_the_first_six_months():
    """A flood entirely inside the excluded window must not reach the wet axes.

    The wet exclusion is the explicit half of the shared six-month metric
    window (the dry axes exclude it through the scored SSI-6 series).
    """
    ref_m, ref_d = _synthetic_monthly(78, seed=0), _synthetic_daily(78, seed=0)
    # Six months from a January 1 start (non-leap): Jan 1 -> Jul 1.
    exclusion_days = 31 + 28 + 31 + 30 + 31 + 30  # 181
    scen_m = np.vstack([_synthetic_monthly(5, seed=1)])
    plain_d = np.vstack([_synthetic_daily(5, seed=1)])
    early_flood_d = np.vstack([_synthetic_daily(5, seed=1, spike=(60, 90))])

    wet = slice(len(hm.DRY_EVENT_METRICS), None)
    H_plain, _ = hm.compute_candidate_hazard_image(
        scen_m, plain_d, ref_m, ref_d, wet_exclusion_days=exclusion_days,
    )
    H_early, _ = hm.compute_candidate_hazard_image(
        scen_m, early_flood_d, ref_m, ref_d, wet_exclusion_days=exclusion_days,
    )
    assert np.allclose(H_early[0, wet], H_plain[0, wet])

    # Without the exclusion the same flood dominates the wet axes.
    H_no_cut, _ = hm.compute_candidate_hazard_image(scen_m, early_flood_d, ref_m, ref_d)
    H_ref, _ = hm.compute_candidate_hazard_image(scen_m, plain_d, ref_m, ref_d)
    assert H_no_cut[0, wet][0] > H_ref[0, wet][0]  # flood_peak_discharge


def test_wet_exclusion_keeps_a_flood_after_the_window():
    """A flood after the excluded window still registers, and the dry axes are
    untouched by the wet exclusion (the monthly input is never truncated)."""
    ref_m, ref_d = _synthetic_monthly(78, seed=0), _synthetic_daily(78, seed=0)
    scen_m = np.vstack([_synthetic_monthly(5, seed=2)])
    plain_d = np.vstack([_synthetic_daily(5, seed=2)])
    late_flood_d = np.vstack([_synthetic_daily(5, seed=2, spike=(900, 930))])

    dry = slice(0, len(hm.DRY_EVENT_METRICS))
    wet = slice(len(hm.DRY_EVENT_METRICS), None)
    H_plain, _ = hm.compute_candidate_hazard_image(
        scen_m, plain_d, ref_m, ref_d, wet_exclusion_days=181,
    )
    H_late, _ = hm.compute_candidate_hazard_image(
        scen_m, late_flood_d, ref_m, ref_d, wet_exclusion_days=181,
    )
    assert H_late[0, wet][0] > H_plain[0, wet][0]
    assert np.allclose(H_late[0, dry], H_plain[0, dry])


def test_wet_exclusion_rejects_a_window_it_would_empty():
    ref_m, ref_d = _synthetic_monthly(78, seed=0), _synthetic_daily(78, seed=0)
    scen_m = np.vstack([_synthetic_monthly(1, seed=3)])
    scen_d = np.vstack([_synthetic_daily(1, seed=3)])
    with pytest.raises(ValueError, match="leaves no daily values"):
        hm.compute_candidate_hazard_image(
            scen_m, scen_d, ref_m, ref_d, wet_exclusion_days=365,
        )


def test_flows_to_series_is_month_start_indexed():
    """SynHydro's SSI needs a DatetimeIndex; the scenario stamp carries the
    scenarios' true December start (the realization epoch). The reference
    keeps its own (January) start — the SSI fit is calendar-month-keyed, so
    the two stamps need not share a month."""
    s = hm.flows_to_series(_synthetic_monthly(2, seed=4))
    assert len(s) == 24
    assert s.index[0].month == 12
    assert s.index[0].month == pd.Timestamp(hm._SCENARIO_STAMP_START).month
    assert pd.Timestamp(hm._REFERENCE_START).month == 1


def test_hazard_image_accepts_month_keyed_reference_start():
    """The SSI fit and transform are keyed by CALENDAR MONTH (spei maps
    observations onto year-2000 dates and groups by month), so a reference
    whose truthful start month differs from the scenario stamp's computes
    finite coordinates rather than raising."""
    ref_m, ref_d = _synthetic_monthly(78, seed=0), _synthetic_daily(78, seed=0)
    scen_m = np.vstack([_synthetic_monthly(5, seed=1)])
    scen_d = np.vstack([_synthetic_daily(5, seed=1)])
    H, axes = hm.compute_candidate_hazard_image(
        scen_m, scen_d, ref_m, ref_d, reference_start="1945-10-01",
    )
    assert H.shape == (1, len(axes))
    assert np.isfinite(H).all()


def test_dry_axes_rotate_with_a_mislabeled_seasonal_reference():
    """Rotation sensitivity: with a strongly seasonal record, mislabeling the
    reference by +3 months must change the dry axes for the same flows.

    This is the class of bug the i.i.d.-gamma fixtures cannot see (their
    monthly gamma fits are identical in every month). Grouping both series by
    TRUE month is what makes the coordinates correct; the assertion is made by
    fitting two references whose only difference is the labeled start month.
    """
    ref = _seasonal_monthly(78, seed=0)
    scen = _seasonal_monthly(6, seed=1)

    calc_true = hm.fit_reference_ssi(ref, start_date="1945-01-01")
    calc_rotated = hm.fit_reference_ssi(ref, start_date="1945-10-01")

    s = hm.flows_to_series(scen)  # January-stamped scenario
    ssi_true = calc_true.transform(s).to_numpy()
    ssi_rotated = calc_rotated.transform(s).to_numpy()

    ok = np.isfinite(ssi_true) & np.isfinite(ssi_rotated)
    assert ok.any()
    # The rotated fit applies each month's gamma to flows from a different
    # season, so the SSI trace must differ materially.
    assert np.nanmax(np.abs(ssi_true[ok] - ssi_rotated[ok])) > 0.5


def test_reference_fit_cache_is_result_preserving():
    """Cache-hit calls must equal a cache-cleared from-scratch computation.

    The second call passes equal-content but distinct-object reference arrays,
    so a hit proves content keying (never id()).
    """
    ref_m, ref_d = _synthetic_monthly(78, seed=0), _synthetic_daily(78, seed=0)
    scen_m = np.vstack([_synthetic_monthly(10, seed=3, dip=(40, 52))])
    scen_d = np.vstack([_synthetic_daily(10, seed=3, spike=(400, 405))])

    hm._REFERENCE_FIT_CACHE.clear()
    H_cold, names_cold = hm.compute_candidate_hazard_image(scen_m, scen_d, ref_m, ref_d)
    assert len(hm._REFERENCE_FIT_CACHE) == 1

    H_warm, names_warm = hm.compute_candidate_hazard_image(
        scen_m.copy(), scen_d.copy(), ref_m.copy(), ref_d.copy()
    )
    assert len(hm._REFERENCE_FIT_CACHE) == 1  # same content -> same key

    hm._REFERENCE_FIT_CACHE.clear()
    H_fresh, _ = hm.compute_candidate_hazard_image(scen_m, scen_d, ref_m, ref_d)

    assert names_cold == names_warm
    np.testing.assert_array_equal(H_cold, H_warm)
    np.testing.assert_array_equal(H_cold, H_fresh)


def test_prefit_kwargs_bypass_and_match_the_cache_path():
    """Passing prefit fits explicitly must reproduce the default path exactly."""
    ref_m, ref_d = _synthetic_monthly(78, seed=0), _synthetic_daily(78, seed=0)
    scen_m = np.vstack([_synthetic_monthly(10, seed=5, dip=(30, 44))])
    scen_d = np.vstack([_synthetic_daily(10, seed=5, spike=(900, 906))])

    dry_calc, threshold, ref_mean = hm.get_reference_fits(ref_m, ref_d)
    H_prefit, _ = hm.compute_candidate_hazard_image(
        scen_m, scen_d, ref_m, ref_d,
        prefit_dry_calc=dry_calc, prefit_threshold=threshold, prefit_ref_mean=ref_mean,
    )
    H_default, _ = hm.compute_candidate_hazard_image(scen_m, scen_d, ref_m, ref_d)
    np.testing.assert_array_equal(H_prefit, H_default)

    with pytest.raises(ValueError, match="prefit_dry_calc requires"):
        hm.compute_candidate_hazard_image(
            scen_m, scen_d, ref_m, ref_d, prefit_dry_calc=dry_calc,
        )


def _ssi_series(values):
    """Monthly SSI series with the given values, preceded and followed by surplus months."""
    vals = [0.5] * 6 + list(values) + [0.5] * 6
    return pd.Series(vals, index=pd.date_range("2000-01-01", periods=len(vals), freq="MS"))


def test_phase_rates_count_elapsed_months_from_the_crossings():
    """Development and termination durations include the crossing month (Parry et al. 2016)."""
    # onset -0.5, minimum -1.5 two months later, last negative month two months after that
    out = hm.critical_event_descriptors(_ssi_series([-0.5, -1.2, -1.5, -0.8, -0.2]))
    assert out["severity"] == pytest.approx(1.5)
    assert out["development_rate"] == pytest.approx(1.5 / 3)
    assert out["termination_rate"] == pytest.approx(1.5 / 3)


def test_phase_rates_when_the_minimum_is_the_first_month():
    out = hm.critical_event_descriptors(_ssi_series([-1.4, -0.6, -0.1]))
    assert out["development_rate"] == pytest.approx(1.4)      # one-month development phase
    assert out["termination_rate"] == pytest.approx(1.4 / 3)


def test_termination_rate_counts_wet_interludes_as_elapsed_time():
    """A short positive interlude inside the event (below the 3-month termination) adds elapsed months."""
    out = hm.critical_event_descriptors(_ssi_series([-1.2, 0.2, -0.3]))
    assert out["duration"] == pytest.approx(2.0)               # negative months only
    assert out["termination_rate"] == pytest.approx(1.2 / 3)   # minimum -> last negative month: 2 + 1


def test_untruncated_event_keeps_the_observed_phase_rates():
    """An event inside the window is flagged on neither side, whatever ssi_pre says."""
    for ssi_pre in (None, -0.5, 0.5):
        out = hm.critical_event_descriptors(
            _ssi_series([-0.5, -1.2, -1.5, -0.8, -0.2]), ssi_pre=ssi_pre
        )
        assert not out["onset_truncated"] and not out["termination_truncated"]
        assert out["development_rate"] == pytest.approx(1.5 / 3)
        assert out["termination_rate"] == pytest.approx(1.5 / 3)
        assert out["event_count"] == 1.0


def test_open_event_at_the_series_end_is_recorded_and_can_control():
    """A qualifying run still below zero in the last month is an event, and the
    controlling one when its deficit is the largest."""
    z = [0.5, -1.2, -0.4, 0.3, 0.3, 0.3, 0.5, -0.6, -1.5, -2.0, -1.1]
    events = hm.drought_events(z)
    assert [(e.start, e.end, e.peak) for e in events] == [(1, 2, 1), (7, 10, 9)]

    out = hm.critical_event_descriptors(z, ssi_pre=0.5)
    assert out["magnitude"] == pytest.approx(5.2)
    assert out["duration"] == 4.0
    assert out["severity"] == pytest.approx(2.0)
    assert not out["onset_truncated"] and out["termination_truncated"]
    assert out["development_rate"] == pytest.approx(2.0 / 3)
    assert out["termination_rate"] == pytest.approx(0.9)       # (-1.1 - -2.0) / 1 month
    assert out["event_count"] == 2.0
    assert out["total_deficit"] == pytest.approx(1.6 + 5.2)


def test_recovered_but_unconfirmed_event_is_recorded_with_an_observed_termination():
    """Two non-negative months before the series ends do not close the event, but
    it is recorded, and its termination (last negative month) is inside the window."""
    z = [0.4, -0.5, -1.3, -0.7, 0.2, 0.1]
    events = hm.drought_events(z)
    assert [(e.start, e.end, e.peak, e.duration) for e in events] == [(1, 3, 2, 3)]

    out = hm.critical_event_descriptors(z, ssi_pre=0.4)
    assert out["magnitude"] == pytest.approx(2.5)
    assert not out["termination_truncated"]
    assert out["termination_rate"] == pytest.approx(1.3 / 2)


def test_truncated_onset_rate_is_the_in_window_decline():
    z = [-0.4, -1.0, -1.6, -0.5, 0.2, 0.2, 0.2]
    for ssi_pre in (-0.3, None, float("nan")):
        out = hm.critical_event_descriptors(z, ssi_pre=ssi_pre)
        assert out["onset_truncated"] and not out["termination_truncated"]
        assert out["development_rate"] == pytest.approx((-0.4 + 1.6) / 2)
        assert out["termination_rate"] == pytest.approx(1.6 / 2)


def test_non_negative_ssi_pre_leaves_an_edge_onset_observed():
    """An event starting at the first scored month after a non-negative month
    crossed zero inside the window: standard development rate."""
    z = [-0.4, -1.0, -1.6, -0.5, 0.2, 0.2, 0.2]
    for ssi_pre in (0.1, 0.0):
        out = hm.critical_event_descriptors(z, ssi_pre=ssi_pre)
        assert not out["onset_truncated"]
        assert out["development_rate"] == pytest.approx(1.6 / 3)


def test_truncated_termination_rate_is_the_in_window_recovery():
    out = hm.critical_event_descriptors([0.3, -0.5, -1.8, -1.2, -0.4], ssi_pre=0.3)
    assert not out["onset_truncated"] and out["termination_truncated"]
    assert out["development_rate"] == pytest.approx(1.8 / 2)
    assert out["termination_rate"] == pytest.approx((-0.4 + 1.8) / 2)


def test_truncated_phase_rates_are_zero_with_the_minimum_on_the_edge():
    onset = hm.critical_event_descriptors([-1.5, -0.8, -0.2, 0.3, 0.3, 0.3])
    assert onset["onset_truncated"]
    assert onset["development_rate"] == 0.0
    assert onset["termination_rate"] == pytest.approx(1.5 / 3)

    end = hm.critical_event_descriptors([0.2, -0.4, -1.1], ssi_pre=0.2)
    assert end["termination_truncated"]
    assert end["termination_rate"] == 0.0
    assert end["development_rate"] == pytest.approx(1.1 / 2)

    both = hm.critical_event_descriptors([-1.2, -0.3, -0.9], ssi_pre=-0.1)
    assert both["onset_truncated"] and both["termination_truncated"]
    assert both["development_rate"] == 0.0
    assert both["termination_rate"] == pytest.approx((-0.9 + 1.2) / 2)


def test_reference_ssi_is_standard_normal_in_every_calendar_month():
    """The two-parameter gamma (location fixed at zero) standardizes every
    calendar month of a near-normal reference record, where a free location
    lets maximum likelihood degenerate."""
    ref = np.random.default_rng(0).normal(100.0, 15.0, size=78 * 12)
    calc = hm.fit_reference_ssi(ref)
    assert all(d.loc == 0.0 for d in calc.fitted_distributions.values())
    ssi = calc.transform(hm.flows_to_series(ref, start_date=hm._REFERENCE_START))
    sd = ssi.groupby(ssi.index.month).std(ddof=0)
    assert len(sd) == 12
    assert ((sd >= 0.9) & (sd <= 1.1)).all()


def test_flow_regime_descriptors_on_a_hand_checkable_series():
    """Two 360-day years of 30-day months: 2.0 then 1.0, a 3-day spike to 5.0 in
    year 1 and a 7-day dip to 0.3 in year 2, normalized by ref_mean = 2."""
    x = np.r_[np.full(360, 2.0), np.full(360, 1.0)]
    x[100:103] = 5.0
    x[450:457] = 0.3
    y1, y2 = 729.0 / 360, 355.1 / 360
    out = hm.flow_regime_descriptors(x, np.full(24, 30), ref_mean=2.0)
    assert out["lowflow_min_12month"] == pytest.approx(y2 / 2)
    assert out["lowflow_min_24month"] == pytest.approx(1084.1 / 720 / 2)
    assert out["lowflow_min_year"] == pytest.approx(y2 / 2)
    assert out["highflow_max_year"] == pytest.approx(y1 / 2)
    assert out["lowflow_min_7day"] == pytest.approx(0.15)
    assert out["flood_max_3day"] == pytest.approx(2.5)
    assert out["annual_cv"] == pytest.approx(abs(y1 - y2) / np.sqrt(2) / ((y1 + y2) / 2))
    assert out["flashiness"] == pytest.approx(8.4 / 1084.1)

    with pytest.raises(ValueError, match="at least 24 months"):
        hm.flow_regime_descriptors(x[:690], np.full(23, 30), ref_mean=2.0)
    with pytest.raises(ValueError, match="at least 24 months"):
        hm.flow_regime_descriptors(x[:-1], np.full(24, 30), ref_mean=2.0)


def test_pot_supplement_counts_runs_and_the_critical_pulse_volume():
    daily = np.array([0.0, 5.0, 6.0, 0.0, 7.0, 9.0, 8.0, 0.0, 4.0])
    out = hm.pot_flood_descriptors(daily, threshold=3.5, ref_mean=2.0)
    assert out["pulse_count"] == 3.0
    assert out["days_above"] == 6.0
    assert out["pulse_duration"] == 3.0
    assert out["pulse_volume"] == pytest.approx((3.5 + 5.5 + 4.5) / 2.0)


def _calendar_scenario(start, n_ffmp_years, *, seed):
    """Daily flows on the true calendar from ``start`` through May of the last
    FFMP year, their monthly means, and the six-month daily cut."""
    t0 = pd.Timestamp(start)
    days = pd.date_range(t0, t0 + pd.DateOffset(months=6 + 12 * n_ffmp_years), freq="D")[:-1]
    daily = pd.Series(np.random.default_rng(seed).gamma(2.0, 100.0, len(days)), index=days)
    cut = int((days < t0 + pd.DateOffset(months=6)).sum())
    return daily.resample("MS").mean().to_numpy(), daily.to_numpy(), cut


def test_supplement_comes_from_the_same_pass_on_the_scored_window():
    """The supplement leaves H untouched, and each column is the named descriptor
    of the scored window, with true month lengths (February 2004 has 29 days)."""
    ref_m, ref_d = _synthetic_monthly(78, seed=0), _synthetic_daily(78, seed=0)
    start = "2002-12-01"
    monthly, daily, cut = _calendar_scenario(start, 3, seed=7)
    args = (monthly[None, :], daily[None, :], ref_m, ref_d)

    H, names = hm.compute_candidate_hazard_image(*args, wet_exclusion_days=cut)
    H_s, names_s, S, s_names = hm.compute_candidate_hazard_image(
        *args, wet_exclusion_days=cut, return_supplement=True, scenario_start=start,
    )
    np.testing.assert_array_equal(H_s, H)
    assert names_s == names
    assert s_names == list(hm.SUPPLEMENT_METRICS)
    assert S.shape == (1, len(hm.SUPPLEMENT_METRICS))

    calc, threshold, ref_mean = hm.get_reference_fits(ref_m, ref_d)
    ssi, ssi_pre = hm.scored_dry_ssi(calc, monthly)
    d = hm.critical_event_descriptors(ssi, ssi_pre=ssi_pre)
    w = hm.pot_flood_descriptors(daily[cut:], threshold=threshold, ref_mean=ref_mean)
    month_days = np.diff(pd.date_range("2003-06-01", "2006-06-01", freq="MS").to_numpy())
    month_days = month_days.astype("timedelta64[D]").astype(int)
    assert month_days[8] == 29
    f = hm.flow_regime_descriptors(daily[cut:], month_days, ref_mean=ref_mean)
    expected = {f"drought_{k}": v for k, v in d.items()}
    expected |= {f"flood_{k}": v for k, v in w.items()} | f
    np.testing.assert_allclose(S[0], [float(expected[n]) for n in hm.SUPPLEMENT_METRICS])


def test_supplement_requires_a_consistent_calendar():
    ref_m, ref_d = _synthetic_monthly(78, seed=0), _synthetic_daily(78, seed=0)
    monthly, daily, cut = _calendar_scenario("2002-12-01", 3, seed=7)
    args = (monthly[None, :], daily[None, :], ref_m, ref_d)
    with pytest.raises(ValueError, match="requires scenario_start"):
        hm.compute_candidate_hazard_image(*args, wet_exclusion_days=cut, return_supplement=True)
    with pytest.raises(ValueError, match="does not span exactly"):
        hm.compute_candidate_hazard_image(
            *args, wet_exclusion_days=cut + 1, return_supplement=True,
            scenario_start="2002-12-01",
        )
    with pytest.raises(ValueError, match="first day of calendar month"):
        hm.compute_candidate_hazard_image(
            *args, wet_exclusion_days=cut, return_supplement=True, scenario_start="2003-01-01",
        )
