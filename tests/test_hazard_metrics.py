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
    assert set(out) == {"duration", "magnitude", "severity", "onset_rate", "recovery_rate"}
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
    assert set(plain) == {"peak_discharge", "pulse_duration", "rise_rate"}
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
    window (the dry axes exclude it implicitly via SSI-6 accumulation).
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
    """SynHydro's SSI needs a DatetimeIndex; the scenario stamp is a true January
    start, matching the reference fit's start month."""
    s = hm.flows_to_series(_synthetic_monthly(2, seed=4))
    assert len(s) == 24
    assert s.index[0].month == 1
    assert s.index[0].month == pd.Timestamp(hm._REFERENCE_START).month


def test_hazard_image_rejects_rotated_reference_start():
    """A reference stamp whose month differs from the scenario stamp must raise:
    the mismatch otherwise degrades to silently NaN/clipped SSI."""
    ref_m, ref_d = _synthetic_monthly(78, seed=0), _synthetic_daily(78, seed=0)
    scen_m = np.vstack([_synthetic_monthly(5, seed=1)])
    scen_d = np.vstack([_synthetic_daily(5, seed=1)])
    with pytest.raises(ValueError, match="share a start month"):
        hm.compute_candidate_hazard_image(
            scen_m, scen_d, ref_m, ref_d, reference_start="1945-10-01",
        )


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
