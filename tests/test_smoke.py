"""Smoke tests for the scengen scaffold.

Covers the parts that are implemented (not stubs): the climate-adjustment transform, the
calendar->water-year reindex, and the global-index determinism contract. Stubbed functions are
expected to raise ``NotImplementedError`` and are asserted as such so the scaffold's surface is
exercised end to end.
"""

from __future__ import annotations

import numpy as np
import pytest

from scengen import diagnostics, forcing_space, hazard_metrics, master_ensemble, subsample
from scengen.manifest import EnsembleManifest


def test_package_imports():
    for mod in (forcing_space, master_ensemble, hazard_metrics, subsample, diagnostics):
        assert mod is not None


def test_climate_adjustment_identity_at_unit_factor():
    """a_j == 1 must be an exact identity on (mean_period, std_period)."""
    rng = np.random.default_rng(0)
    y_bar = rng.normal(size=12)
    sigma = np.abs(rng.normal(size=12)) + 0.1
    a = np.ones(12)
    y_new, s_new = forcing_space.apply_climate_adjustment(y_bar, sigma, a)
    np.testing.assert_allclose(y_new, y_bar, atol=1e-12)
    np.testing.assert_allclose(s_new, sigma, atol=1e-12)


def test_climate_adjustment_increases_mean_with_factor():
    """A wetter factor (a>1) must raise the real-space mean exp(Y_bar + sigma^2/2)."""
    y_bar = np.zeros(12)
    sigma = np.full(12, 0.3)
    a = np.full(12, 1.2)
    y_new, s_new = forcing_space.apply_climate_adjustment(y_bar, sigma, a)
    real_mean_old = np.exp(y_bar + sigma**2 / 2)
    real_mean_new = np.exp(y_new + s_new**2 / 2)
    np.testing.assert_allclose(real_mean_new / real_mean_old, a, rtol=1e-10)


def test_calendar_to_water_year_reorders():
    cal = np.arange(1, 13, dtype=float)  # Jan..Dec
    wy = forcing_space.calendar_to_water_year(cal)
    assert list(wy) == [10, 11, 12, 1, 2, 3, 4, 5, 6, 7, 8, 9]


def test_child_seed_global_index_invariance():
    """The determinism contract: child k is identical regardless of total N or partitioning."""
    seed = 1234
    small = master_ensemble.child_seed_sequence(seed, 10)
    large = master_ensemble.child_seed_sequence(seed, 100)
    for k in (0, 3, 9):
        np.testing.assert_array_equal(
            small[k].generate_state(8), large[k].generate_state(8)
        )


def test_manifest_roundtrip(tmp_path):
    m = EnsembleManifest(
        design="hazard_filling",
        draw=0,
        n_realizations=200,
        realization_years=5,
        master_seed=42,
        realization_global_indices=[1, 7, 42],
        forcing_hash="deadbeef",
        hazard_axes=["min_ssi3_neg", "total_flow_neg", "summer_recession"],
        slug="hazard_filling_kn5yr_n200_d0",
    )
    path = m.write(tmp_path)
    back = EnsembleManifest.from_json(path)
    assert back.design == "hazard_filling"
    assert tuple(back.realization_global_indices) == (1, 7, 42)


@pytest.mark.parametrize(
    "call",
    [
        # Still-stubbed surfaces (subsample, hazard_metrics, and the coverage
        # diagnostics are now implemented and covered by their own tests).
        lambda: subsample.support_point_subsample(np.zeros((10, 3)), 5, seed=0),
    ],
)
def test_stubs_raise(call):
    with pytest.raises(NotImplementedError):
        call()
