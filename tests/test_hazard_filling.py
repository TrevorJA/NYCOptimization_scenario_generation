"""Tests for the hazard-filling driver. Skipped when SynHydro is unavailable."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("synhydro", reason="hazard_filling driver needs SynHydro's SSI")

from scengen import hazard_filling as hf  # noqa: E402
from scengen import hazard_metrics as hm  # noqa: E402


def _monthly(n_years, *, seed, scale=100.0):
    rng = np.random.default_rng(seed)
    return rng.gamma(shape=2.0, scale=scale, size=n_years * 12)


def _daily(n_years, *, seed, scale=100.0):
    rng = np.random.default_rng(seed + 1000)
    return rng.gamma(shape=2.0, scale=scale, size=n_years * 365)


def _candidate_image(n_scen=60):
    """Candidate hazard image for a toy pool (the streamed `hazard_image.npz` payload)."""
    ref_m = _monthly(78, seed=0)
    ref_d = _daily(78, seed=0)
    scen_m = np.vstack([_monthly(5, seed=s) for s in range(n_scen)])
    scen_d = np.vstack([_daily(5, seed=s) for s in range(n_scen)])
    return hm.compute_candidate_hazard_image(scen_m, scen_d, ref_m, ref_d)


def test_select_from_candidate_image_basic():
    """The live entry point: screen a precomputed candidate image, then select."""
    H, axes = _candidate_image(n_scen=60)
    assert axes == list(hm.CANDIDATE_EVENT_METRICS)
    assert H.shape == (60, 8)

    out = hf.select_from_candidate_image(H, axes, n=10, seed=0)

    assert out["selected_rows"].shape == (10,)
    assert set(out["selected_rows"].tolist()) <= set(range(60))
    assert out["candidate_axes"] == list(hm.CANDIDATE_EVENT_METRICS)
    assert out["H_candidates"].shape == (60, 8)
    # tail-balanced chosen set: <=2 dry + <=2 wet (<=4 total), subset of candidates.
    assert 1 <= len(out["chosen_axes"]) <= 4
    assert set(out["chosen_axes"]) <= set(hm.CANDIDATE_EVENT_METRICS)
    assert sum(a.startswith("drought") for a in out["chosen_axes"]) <= 2
    assert sum(a.startswith("flood") for a in out["chosen_axes"]) <= 2
    assert {"selected_L2_star", "random_L2_star"} <= set(out["coverage"])


def test_select_from_candidate_image_is_deterministic_given_seed():
    """LHS + NN-snap has no stochastic search: the same seed must reproduce the design."""
    H, axes = _candidate_image(n_scen=60)
    a = hf.select_from_candidate_image(H, axes, n=10, seed=3)["selected_rows"]
    b = hf.select_from_candidate_image(H, axes, n=10, seed=3)["selected_rows"]
    np.testing.assert_array_equal(a, b)


def test_select_from_candidate_image_absolute_space():
    """The retained non-campaign sensitivity: selection in absolute magnitude space."""
    H, axes = _candidate_image(n_scen=60)
    out = hf.select_from_candidate_image(H, axes, n=10, seed=0, selector_space="abs")
    assert out["selected_rows"].shape == (10,)


def test_select_from_candidate_image_rejects_unknown_space():
    H, axes = _candidate_image(n_scen=60)
    with pytest.raises(ValueError):
        hf.select_from_candidate_image(H, axes, n=10, seed=0, selector_space="nope")
