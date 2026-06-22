"""Tests for the hazard-filling subsample selector (pure numpy/scipy; no SSI)."""

from __future__ import annotations

import numpy as np
import pytest

from scengen import subsample


def _clustered_hazard_image(M=400, d=3, seed=0):
    """A skewed, clustered hazard image: most mass in a blob, a few extremes."""
    rng = np.random.default_rng(seed)
    n_tail = max(1, M // 10)
    blob = rng.normal(0.0, 1.0, size=(M - n_tail, d))
    tail = rng.normal(5.0, 2.0, size=(n_tail, d))  # rare severe corner
    return np.vstack([blob, tail])


def test_empirical_cdf_normalize_is_uniform_per_axis():
    H = _clustered_hazard_image()
    X = subsample.empirical_cdf_normalize(H)
    assert X.shape == H.shape
    assert X.min() > 0.0 and X.max() <= 1.0 + 1e-12
    # Each axis is approximately uniform: mean ~ 0.5.
    assert np.allclose(X.mean(axis=0), 0.5, atol=0.05)


def test_subsample_returns_sorted_unique_subset():
    H = _clustered_hazard_image()
    sel = subsample.hazard_filling_subsample(H, 20, seed=1, iters=2000)
    assert sel.shape == (20,)
    assert len(set(sel.tolist())) == 20
    assert np.all(np.diff(sel) > 0)              # sorted, unique
    assert set(sel.tolist()) <= set(range(len(H)))


def test_random_subsample_basic():
    H = _clustered_hazard_image()
    sel = subsample.random_subsample(H, 15, seed=3)
    assert sel.shape == (15,)
    assert len(set(sel.tolist())) == 15


def test_hazard_filling_beats_random_on_discrepancy():
    """Core claim: the selector covers hazard space more uniformly than a random draw.

    Measured as lower L2-star discrepancy in the empirical-CDF-normalized space,
    averaged over several seeds (the build-QC check of methods 6a).
    """
    H = _clustered_hazard_image(M=400, d=3, seed=7)
    X = subsample.empirical_cdf_normalize(H)
    lb, ub = np.zeros(3), np.ones(3)
    n = 24

    def disc(sel):
        return subsample.coverage_metrics(X[sel], lb, ub)["L2_star_discrepancy"]

    hf = np.mean([
        disc(subsample.hazard_filling_subsample(H, n, seed=s, iters=6000))
        for s in range(4)
    ])
    rand = np.mean([disc(subsample.random_subsample(H, n, seed=s)) for s in range(4)])
    assert hf < rand, f"hazard-filling discrepancy {hf:.4g} !< random {rand:.4g}"


def test_subsample_n_equals_m_returns_all():
    H = _clustered_hazard_image(M=12, d=2)
    sel = subsample.hazard_filling_subsample(H, 12, seed=0, iters=100)
    assert np.array_equal(sel, np.arange(12))


def test_subsample_too_large_raises():
    H = _clustered_hazard_image(M=10, d=2)
    with pytest.raises(ValueError):
        subsample.hazard_filling_subsample(H, 11, seed=0)
