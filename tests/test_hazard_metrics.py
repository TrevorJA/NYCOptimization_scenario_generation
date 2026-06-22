"""Tests for the copied SSI hazard metrics. Skipped when SynHydro is unavailable."""

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


def test_primary_metrics_shape_and_finiteness():
    ref = _synthetic_monthly(78, seed=0)
    calc = hm.fit_reference_ssi(ref)
    scenario = _synthetic_monthly(5, seed=1)
    out = hm.compute_metrics_for_scenario(scenario, calc)
    assert set(out) == set(hm.PRIMARY_METRICS)
    assert all(np.isfinite(v) for v in out.values())
    assert 0.0 <= out["time_in_drought_fraction"] <= 1.0


def test_hazard_image_shape():
    ref = _synthetic_monthly(78, seed=0)
    scenarios = np.vstack([_synthetic_monthly(5, seed=s) for s in range(8)])
    H, names = hm.compute_hazard_image(scenarios, ref)
    assert H.shape == (8, 3)
    assert names == list(hm.PRIMARY_METRICS)
    assert np.isfinite(H).all()


def test_drought_scenario_has_more_time_in_drought():
    """A scenario with an imposed low-flow dip should spend more time in drought."""
    ref = _synthetic_monthly(78, seed=0)
    calc = hm.fit_reference_ssi(ref)
    normal = hm.compute_metrics_for_scenario(_synthetic_monthly(5, seed=2), calc)
    drought = hm.compute_metrics_for_scenario(
        _synthetic_monthly(5, seed=2, dip=(24, 36)), calc
    )
    assert drought["time_in_drought_fraction"] >= normal["time_in_drought_fraction"]


def test_custom_metric_subset():
    ref = _synthetic_monthly(78, seed=0)
    calc = hm.fit_reference_ssi(ref)
    out = hm.compute_metrics_for_scenario(
        _synthetic_monthly(5, seed=3), calc,
        metric_names=("frequency", "max_duration"),
    )
    assert set(out) == {"frequency", "max_duration"}
