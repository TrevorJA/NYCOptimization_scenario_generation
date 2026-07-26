"""Tests for the candidate event-descriptor hazard image. Skipped without SynHydro."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("synhydro", reason="hazard_metrics needs SynHydro's SSI")

from scengen import hazard_metrics as hm  # noqa: E402


def _synthetic_monthly(n_years, *, seed, scale=100.0, dip=None):
    """Gamma-distributed monthly flows (Oct-start water years), optional drought dip."""
    rng = np.random.default_rng(seed)
    flows = rng.gamma(shape=2.0, scale=scale, size=n_years * 12)
    if dip is not None:
        lo, hi = dip
        flows[lo:hi] *= 0.2  # impose a multi-month low-flow event
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
    assert set(out) == {"duration", "volume", "depth", "onset_rate", "recovery_rate"}
    assert all(np.isfinite(v) and v >= 0.0 for v in out.values())


def test_drought_scenario_has_larger_deficit_volume():
    """An imposed multi-month low-flow dip must deepen the controlling event."""
    ref = _synthetic_monthly(78, seed=0)
    calc = hm.fit_reference_ssi(ref)

    def volume(flows):
        return hm.critical_event_descriptors(
            calc.transform(hm.flows_to_series(flows))
        )["volume"]

    normal = volume(_synthetic_monthly(5, seed=2))
    drought = volume(_synthetic_monthly(5, seed=2, dip=(24, 36)))
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
    assert set(plain) == {"peak_magnitude", "pulse_duration", "rise_rate"}
    assert spiked["peak_magnitude"] > plain["peak_magnitude"]
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
    exclusion_days = 182  # Oct 1 -> Apr 1 in a non-leap water year
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
    assert H_no_cut[0, wet][0] > H_ref[0, wet][0]  # flood_peak_magnitude


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
        scen_m, plain_d, ref_m, ref_d, wet_exclusion_days=182,
    )
    H_late, _ = hm.compute_candidate_hazard_image(
        scen_m, late_flood_d, ref_m, ref_d, wet_exclusion_days=182,
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
    """SynHydro's SSI needs a DatetimeIndex; October start aligns the water year."""
    s = hm.flows_to_series(_synthetic_monthly(2, seed=4))
    assert len(s) == 24
    assert s.index[0].month == 10
