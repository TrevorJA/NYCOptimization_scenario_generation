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
    sel = subsample.cdf_filling_subsample(H, 20, seed=1)
    assert sel.shape == (20,)
    assert len(set(sel.tolist())) == 20
    assert np.all(np.diff(sel) > 0)              # sorted, unique
    assert set(sel.tolist()) <= set(range(len(H)))


def test_random_subsample_basic():
    H = _clustered_hazard_image()
    sel = subsample.random_subsample(H, 15, seed=3)
    assert sel.shape == (15,)
    assert len(set(sel.tolist())) == 15


def test_cdf_filling_beats_random_on_discrepancy():
    """The rank-space sensitivity covers rank space more uniformly than a random draw.

    Measured as lower L2-star discrepancy in the empirical-CDF-normalized space,
    averaged over several seeds (the independent build-QC check).
    """
    H = _clustered_hazard_image(M=400, d=3, seed=7)
    X = subsample.empirical_cdf_normalize(H)
    lb, ub = np.zeros(3), np.ones(3)
    n = 24

    def disc(sel):
        return subsample.coverage_metrics(X[sel], lb, ub)["L2_star_discrepancy"]

    cdf = np.mean([
        disc(subsample.cdf_filling_subsample(H, n, seed=s))
        for s in range(4)
    ])
    rand = np.mean([disc(subsample.random_subsample(H, n, seed=s)) for s in range(4)])
    assert cdf < rand, f"cdf-filling discrepancy {cdf:.4g} !< random {rand:.4g}"


def test_absolute_filling_subsample_returns_sorted_unique_subset():
    H = _clustered_hazard_image(M=300, d=3, seed=11)
    sel = subsample.absolute_filling_subsample(H, 24, seed=2)
    assert sel.shape == (24,)
    assert len(set(sel.tolist())) == 24
    assert np.all(np.diff(sel) > 0)
    assert set(sel.tolist()) <= set(range(len(H)))


def test_absolute_filling_beats_random_in_absolute_space():
    """In its OWN (magnitude) space, absolute filling covers more uniformly than random."""
    H = _clustered_hazard_image(M=300, d=3, seed=5)
    X = subsample.minmax_normalize(H)
    lb, ub = np.zeros(3), np.ones(3)
    n = 24

    def disc(sel):
        return subsample.coverage_metrics(X[sel], lb, ub)["L2_star_discrepancy"]

    absf = np.mean([disc(subsample.absolute_filling_subsample(H, n, seed=s)) for s in range(4)])
    rand = np.mean([disc(subsample.random_subsample(H, n, seed=s)) for s in range(4)])
    assert absf < rand


def test_absolute_filling_weights_tails_more_than_cdf_filling():
    """The campaign selector lands more mass in the sparse upper tail than the sensitivity.

    Uniform-in-magnitude filling over a skewed pool over-represents the rare
    extremes relative to their frequency -- the deliberate distribution shift the
    study tests; uniform-in-rank filling does not.
    """
    H = _clustered_hazard_image(M=400, d=3, seed=7)
    p90 = np.percentile(H, 90, axis=0)

    def tail_share(sel):
        return float(np.mean((H[sel] > p90).any(axis=1)))

    n = 24
    absf = np.mean([tail_share(subsample.absolute_filling_subsample(H, n, seed=s)) for s in range(4)])
    cdf = np.mean([tail_share(subsample.cdf_filling_subsample(H, n, seed=s)) for s in range(4)])
    assert absf > cdf


def test_minmax_normalize_robust_bounds_resist_outliers():
    """One extreme outlier must not compress the bulk of the pool.

    With the robust p1/p99 campaign bounds the outlier clips to the box face and
    the bulk keeps its spread; with full-range (0/100) bounds the outlier owns
    the box and the bulk collapses into a sliver. This is the geometry-stability
    property the campaign normalization exists for.
    """
    rng = np.random.default_rng(0)
    H = rng.uniform(0.0, 1.0, size=(500, 2))
    H[0, 0] = 1e4  # single wild outlier on axis 0
    bulk = np.arange(1, 500)

    robust = subsample.minmax_normalize(H)  # campaign default p1/p99
    full = subsample.minmax_normalize(H, lo_pct=0.0, hi_pct=100.0)
    assert robust[0, 0] == 1.0  # outlier clipped to the face, still selectable
    assert np.ptp(robust[bulk, 0]) > 0.9  # bulk spread preserved
    assert np.ptp(full[bulk, 0]) < 0.01  # full-range: bulk collapses


def test_robust_range_bounds_match_percentiles():
    rng = np.random.default_rng(1)
    H = rng.gamma(2.0, 1.0, size=(1000, 3))
    lo, hi = subsample.robust_range_bounds(H)
    assert np.allclose(lo, np.percentile(H, subsample.ROBUST_LO_PCT, axis=0))
    assert np.allclose(hi, np.percentile(H, subsample.ROBUST_HI_PCT, axis=0))
    assert np.all(hi > lo)


def test_zero_inflated_axis_gets_natural_zero_lower_bound():
    """On a zero-inflated (dry event) axis, p1 collapses to the natural zero."""
    rng = np.random.default_rng(2)
    col = rng.gamma(2.0, 1.0, size=1000)
    col[:300] = 0.0  # 30% zero-event windows
    H = np.column_stack([col, rng.uniform(1.0, 5.0, size=1000)])
    lo, _ = subsample.robust_range_bounds(H)
    assert lo[0] == 0.0


def test_subsample_n_equals_m_returns_all():
    H = _clustered_hazard_image(M=12, d=2)
    sel = subsample.cdf_filling_subsample(H, 12, seed=0)
    assert np.array_equal(sel, np.arange(12))


def test_subsample_too_large_raises():
    H = _clustered_hazard_image(M=10, d=2)
    with pytest.raises(ValueError):
        subsample.cdf_filling_subsample(H, 11, seed=0)
