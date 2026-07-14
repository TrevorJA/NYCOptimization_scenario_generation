"""Tests for the forcing-space envelope loading, sampling, and climate transform."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scengen import forcing_space as fs


# ---------------------------------------------------------------------------
# Synthetic CMIP6 tables (no dependency on the external CMIP6 checkout)
# ---------------------------------------------------------------------------

def _write_mean_frac_csv(path):
    """A `_frac_` table: one historical col, one baseline ssp col, two future ssp cols."""
    months = np.arange(1, 13)
    df = pd.DataFrame({"month": months})
    df["nhmv10_withObsScaled"] = 1.0  # pure-historical (no ssp token) -> dropped
    df["PRMS_RAPID_ACCESS-CM2_ssp126_r1_DBCCA_1980_2019"] = 1.0  # baseline -> dropped by default
    df["PRMS_RAPID_ACCESS-CM2_ssp245_r1_DBCCA_2060_2099"] = np.linspace(0.7, 1.3, 12)
    df["PRMS_RAPID_ACCESS-CM2_ssp370_r1_DBCCA_2060_2099"] = np.linspace(1.2, 0.8, 12)
    df.to_csv(path, index=False)
    return path


def _write_abs_csv(path, scale):
    months = np.arange(1, 13)
    df = pd.DataFrame({"month": months})
    df["PRMS_RAPID_ACCESS-CM2_ssp126_r1_DBCCA_1980_2019"] = scale * np.full(12, 100.0)
    df["PRMS_RAPID_ACCESS-CM2_ssp245_r1_DBCCA_2060_2099"] = scale * np.full(12, 120.0)
    df.to_csv(path, index=False)
    return path


def test_load_cmip6_envelope_shape_order_and_filtering(tmp_path):
    csv = _write_mean_frac_csv(tmp_path / "frac.csv")
    env = fs.load_cmip6_envelope(csv)
    # Historical + baseline (1980_2019) columns dropped; only the two futures remain.
    assert list(env.columns) == [
        "PRMS_RAPID_ACCESS-CM2_ssp245_r1_DBCCA_2060_2099",
        "PRMS_RAPID_ACCESS-CM2_ssp370_r1_DBCCA_2060_2099",
    ]
    # Rows reindexed to water-year order.
    assert list(env.index) == list(fs.WATER_YEAR_MONTHS)
    assert env.shape == (12, 2)


def test_load_cmip6_envelope_rejects_non_frac(tmp_path):
    months = np.arange(1, 13)
    df = pd.DataFrame({"month": months, "X_ssp245_2060_2099": np.full(12, 40.0)})  # percent-like
    csv = tmp_path / "prc.csv"
    df.to_csv(csv, index=False)
    with pytest.raises(ValueError):
        fs.load_cmip6_envelope(csv)


def test_derive_variance_envelope_matches_manual(tmp_path):
    mean_csv = _write_abs_csv(tmp_path / "means.csv", scale=1.0)
    std_csv = _write_abs_csv(tmp_path / "stds.csv", scale=1.0)
    # std identical structure to mean -> std_chg == mean_chg -> v == 1 everywhere.
    cv = fs.derive_variance_envelope(mean_csv, std_csv)
    assert cv.shape == (12, 1)
    assert list(cv.index) == list(fs.WATER_YEAR_MONTHS)
    np.testing.assert_allclose(cv.values, 1.0, rtol=1e-12)


def test_water_year_calendar_roundtrip():
    cal = np.arange(1, 13, dtype=float)
    wy = fs.calendar_to_water_year(cal)
    back = fs.water_year_to_calendar(wy)
    np.testing.assert_array_equal(back, cal)


# ---------------------------------------------------------------------------
# (harmonic sampler tests below)
# ---------------------------------------------------------------------------

def _toy_envelope(seed=0, K=20):
    """(12, K) anchor cloud of plausible multipliers with seasonal structure."""
    rng = np.random.default_rng(seed)
    base = 1.0 + 0.25 * np.sin(np.linspace(0, 2 * np.pi, 12))[:, None]
    return base + 0.15 * rng.standard_normal((12, K))


# ---------------------------------------------------------------------------
# Harmonic parameterization + CMIP6-based hypercube sampler
# ---------------------------------------------------------------------------

def test_fit_reconstruct_roundtrip_exact_for_band_limited():
    t = np.arange(12)
    w = 2 * np.pi / 12
    a1 = np.exp(0.10 + 0.20 * np.cos(w * t - 0.5) + 0.10 * np.cos(2 * w * t - 1.0))
    a2 = np.exp(-0.05 + 0.15 * np.cos(w * t - 0.3) + 0.04 * np.cos(2 * w * t + 0.7))
    env = np.column_stack([a1, a2])  # (12, 2)
    fit = fs.fit_harmonic_params(env, order=2)
    assert fit["m"].shape == (2,) and fit["amp"].shape == (2, 2) and fit["phase"].shape == (2, 2)
    params = np.column_stack([fit["m"], fit["amp"][:, 0], fit["phase"][:, 0],
                              fit["amp"][:, 1], fit["phase"][:, 1]])
    rec = fs.reconstruct_harmonic(params, order=2)
    np.testing.assert_allclose(rec, env.T, rtol=1e-8)  # band-limited -> exact


def test_harmonic_param_box_is_empirical_90pct_range():
    env = _toy_envelope()
    fit = fs.fit_harmonic_params(env, order=2)
    lo, hi, names = fs.harmonic_param_box(fit)  # default: empirical 90% range (p5-p95)
    assert names[:3] == ["m", "r1", "psi1"]
    m, r1 = np.asarray(fit["m"]), fit["amp"][:, 0]
    np.testing.assert_allclose([lo[0], hi[0]], [np.percentile(m, 5), np.percentile(m, 95)], atol=1e-9)
    np.testing.assert_allclose([lo[1], hi[1]],
                               [max(0.0, np.percentile(r1, 5)), np.percentile(r1, 95)], atol=1e-9)
    # the 90% range is strictly inside the full min/max
    assert lo[0] > m.min() - 1e-9 and hi[0] < m.max() + 1e-9
    assert np.all(np.array(lo)[[1, 3]] >= 0.0)  # amplitude lower bounds floored at 0


def test_sample_harmonic_forcing_shape_positive_deterministic():
    env = _toy_envelope()
    a, params, names = fs.sample_harmonic_forcing(200, env, seed=1, return_params=True)
    assert a.shape == (200, 12) and np.all(a > 0.0)
    assert params.shape == (200, len(names))
    assert names == ["m", "r1", "r2"]  # default fix_phase samples amplitudes only
    a2 = fs.sample_harmonic_forcing(200, env, seed=1)
    np.testing.assert_array_equal(a, a2)  # deterministic given seed
    a3 = fs.sample_harmonic_forcing(200, env, seed=2)
    assert not np.allclose(a, a3)


def test_sample_harmonic_forcing_full_box_inside():
    env = _toy_envelope()
    fit = fs.fit_harmonic_params(env, order=2)
    lo, hi, _ = fs.harmonic_param_box(fit, margin=0.1)
    _, params, _ = fs.sample_harmonic_forcing(300, env, seed=0, margin=0.1, fix_phase=False, return_params=True)
    assert params.shape[1] == 5
    assert np.all(params >= lo - 1e-9) and np.all(params <= hi + 1e-9)


def test_fixed_phase_preserves_seasonal_shape():
    """Quinn-style fixed-phase sampling anchors every profile to the canonical CMIP6 shape.

    Anchors have a VARYING annual phase (peak month spread over a season). Independent phase sampling
    spreads the sampled peak month across that range; fixing the phase at the canonical (mean) shape
    clusters every profile near the canonical peak -> much smaller peak-month spread.
    """
    t = np.arange(12)
    w = 2 * np.pi / 12
    rng = np.random.default_rng(0)
    anchors = []
    for _ in range(40):
        peak_k = 3 + rng.uniform(-1.5, 1.5)        # annual peak month varies across anchors
        r1 = 0.18 + 0.10 * rng.random()
        anchors.append(np.exp(0.05 + r1 * np.cos(w * (t - peak_k)) + 0.03 * np.cos(2 * w * (t - 3))))
    env = np.column_stack(anchors)  # (12, 40)
    peak = lambda A: np.array([np.argmax(A[k]) for k in range(len(A))])
    std_fix = peak(fs.sample_harmonic_forcing(600, env, seed=1, fix_phase=True)).std()
    std_ind = peak(fs.sample_harmonic_forcing(600, env, seed=1, fix_phase=False)).std()
    assert std_fix < std_ind - 0.3   # fixed-phase clusters peaks far more tightly
    assert std_fix < 0.8


# ---------------------------------------------------------------------------
# Sampling method: LHS vs i.i.d. (the distributional-equivalence control)
# ---------------------------------------------------------------------------

def _amp_box(env, *, order=2, margin=0.0):
    """Per-axis (lo, hi) of the sampled amplitude box [m, r1, r2, ...] (fix_phase geometry)."""
    fit = fs.fit_harmonic_params(env, order=order)
    lo, hi, _ = fs.harmonic_param_box(fit, margin=margin)
    amp_idx = [0] + [1 + 2 * h for h in range(order)]
    return lo[amp_idx], hi[amp_idx]


def _stratification_occupancy(params, lo, hi):
    """Fraction of the n per-axis equal-probability strata that are occupied, averaged over axes.

    An n-point Latin hypercube puts exactly one point in each of the n strata of every axis -> 1.0.
    An n-point i.i.d. sample leaves strata empty by the coupon-collector argument -> ~1 - 1/e.
    """
    n = params.shape[0]
    u = (params - lo) / (hi - lo)
    bins = np.clip((u * n).astype(int), 0, n - 1)
    return float(np.mean([len(np.unique(bins[:, a])) / n for a in range(params.shape[1])]))


@pytest.mark.parametrize("method", ["lhs", "iid"])
def test_sample_harmonic_forcing_method_shape_bounds_deterministic(method):
    env = _toy_envelope()
    lo, hi = _amp_box(env)
    a, params, names = fs.sample_harmonic_forcing(
        200, env, seed=3, method=method, return_params=True
    )
    assert a.shape == (200, 12) and np.all(a > 0.0)
    assert names == ["m", "r1", "r2"]  # intrinsic coords; phases are fixed, not free
    assert params.shape == (200, 3)
    assert np.all(params >= lo - 1e-9) and np.all(params <= hi + 1e-9)
    # deterministic in the seed, and the seed actually matters
    np.testing.assert_array_equal(a, fs.sample_harmonic_forcing(200, env, seed=3, method=method))
    assert not np.allclose(a, fs.sample_harmonic_forcing(200, env, seed=4, method=method))


def test_sample_harmonic_forcing_rejects_unknown_method():
    env = _toy_envelope()
    with pytest.raises(ValueError, match="unknown sampling method"):
        fs.sample_harmonic_forcing(10, env, seed=0, method="sobol")


def test_lhs_and_iid_differ():
    env = _toy_envelope()
    p_lhs = fs.sample_harmonic_forcing(100, env, seed=7, method="lhs", return_params=True)[1]
    p_iid = fs.sample_harmonic_forcing(100, env, seed=7, method="iid", return_params=True)[1]
    assert not np.allclose(p_lhs, p_iid)


def test_iid_is_uniform_over_the_box_and_lhs_is_detectably_more_regular():
    """Guards the control: "iid" must be plain Monte Carlo, NOT a stratified design.

    The scenario-design control rests on a uniform random size-N subset of an i.i.d. pool being
    distributionally identical to N i.i.d. draws. A random subset of an LHS pool is not — LHS
    imposes per-axis stratification. This test makes that distinction *tested*, not just documented:
    the i.i.d. draw's marginals are uniform over the box (KS, generous alpha) but its per-axis
    stratification is far from perfect, while LHS fills every stratum exactly.
    """
    from scipy.stats import kstest

    env = _toy_envelope()
    n = 4000
    lo, hi = _amp_box(env)
    p_iid = fs.sample_harmonic_forcing(n, env, seed=11, method="iid", return_params=True)[1]
    p_lhs = fs.sample_harmonic_forcing(n, env, seed=11, method="lhs", return_params=True)[1]

    # (1) i.i.d. marginals are consistent with uniform over each axis' box.
    for a in range(p_iid.shape[1]):
        p = kstest(p_iid[:, a], "uniform", args=(lo[a], hi[a] - lo[a])).pvalue
        assert p > 1e-3, f"axis {a}: i.i.d. marginal not uniform over the box (KS p={p:.2g})"

    # (2) LHS is near-perfectly stratified; the i.i.d. sample is demonstrably not.
    occ_lhs = _stratification_occupancy(p_lhs, lo, hi)
    occ_iid = _stratification_occupancy(p_iid, lo, hi)
    assert occ_lhs > 0.99, f"LHS should occupy every stratum (got {occ_lhs:.3f})"
    assert occ_iid == pytest.approx(1.0 - 1.0 / np.e, abs=0.05)  # coupon-collector, ~0.632
    assert occ_lhs - occ_iid > 0.25  # the two designs are detectably different


def test_method_honored_in_full_box_branch():
    """`fix_phase=False` (all [m, r1, psi1, ...] free) routes through the same method switch."""
    env = _toy_envelope()
    fit = fs.fit_harmonic_params(env, order=2)
    lo, hi, names = fs.harmonic_param_box(fit)
    n = 1000
    p_iid = fs.sample_harmonic_forcing(
        n, env, seed=5, fix_phase=False, method="iid", return_params=True
    )[1]
    p_lhs = fs.sample_harmonic_forcing(
        n, env, seed=5, fix_phase=False, method="lhs", return_params=True
    )[1]
    assert p_iid.shape == (n, len(names)) == p_lhs.shape
    assert np.all(p_iid >= lo - 1e-9) and np.all(p_iid <= hi + 1e-9)
    assert _stratification_occupancy(p_lhs, lo, hi) > 0.99
    assert _stratification_occupancy(p_iid, lo, hi) < 0.9
    with pytest.raises(ValueError):
        fs.sample_harmonic_forcing(10, env, seed=0, fix_phase=False, method="nope")


# ---------------------------------------------------------------------------
# apply_climate_adjustment (c_j conventions)
# ---------------------------------------------------------------------------

def _baseline_stats(n_sites=None, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.normal(size=12)
    s = np.abs(rng.normal(size=12)) + 0.2
    if n_sites is not None:
        y = np.repeat(y[:, None], n_sites, axis=1)
        s = np.repeat(s[:, None], n_sites, axis=1)
    return y, s


def test_cv_preserving_keeps_log_sigma():
    """Default (c=a) is CV-preserving -> log-space sigma is exactly unchanged."""
    y, s = _baseline_stats()
    a = np.linspace(0.8, 1.3, 12)
    _, s_new = fs.apply_climate_adjustment(y, s, a)  # default c=a
    np.testing.assert_allclose(s_new, s, atol=1e-12)


def test_absolute_sd_preserved():
    """c=1 keeps the real-space SD fixed while scaling the mean by a."""
    y, s = _baseline_stats()
    a = np.full(12, 1.4)
    y_new, s_new = fs.apply_climate_adjustment(y, s, a, preserve_absolute_sd=True)
    real_sd_old = np.exp(y + s**2 / 2) * np.sqrt(np.exp(s**2) - 1)
    real_sd_new = np.exp(y_new + s_new**2 / 2) * np.sqrt(np.exp(s_new**2) - 1)
    np.testing.assert_allclose(real_sd_new, real_sd_old, rtol=1e-10)


def test_independent_variance_scales_cv_by_v():
    """c = a * v scales the real-space CV by exactly v."""
    y, s = _baseline_stats()
    a = np.full(12, 1.2)
    v = np.linspace(0.8, 1.5, 12)
    y_new, s_new = fs.apply_climate_adjustment(y, s, a, c_profile=a * v)
    cv_old = np.sqrt(np.exp(s**2) - 1)
    cv_new = np.sqrt(np.exp(s_new**2) - 1)
    np.testing.assert_allclose(cv_new / cv_old, v, rtol=1e-10)


def test_apply_broadcasts_over_sites():
    y2, s2 = _baseline_stats(n_sites=4)
    a = np.linspace(0.9, 1.2, 12)
    v = np.linspace(0.9, 1.1, 12)
    y_new, s_new = fs.apply_climate_adjustment(y2, s2, a, c_profile=a * v)
    assert y_new.shape == (12, 4) and s_new.shape == (12, 4)
    # Every site column is adjusted identically (per-month factor broadcast).
    assert np.allclose(y_new, y_new[:, [0]], atol=1e-12)


def test_real_mean_scales_by_a_regardless_of_c():
    y, s = _baseline_stats()
    a = np.full(12, 1.3)
    for kwargs in ({}, {"preserve_absolute_sd": True}, {"c_profile": a * 1.4}):
        y_new, s_new = fs.apply_climate_adjustment(y, s, a, **kwargs)
        ratio = np.exp(y_new + s_new**2 / 2) / np.exp(y + s**2 / 2)
        np.testing.assert_allclose(ratio, a, rtol=1e-10)


def test_forcing_hash_stable_and_sensitive():
    p = np.ones((5, 12))
    h1 = fs.forcing_hash(p, envelope_csv="a.csv", margin=0.075, seed=0)
    h2 = fs.forcing_hash(p, envelope_csv="a.csv", margin=0.075, seed=0)
    h3 = fs.forcing_hash(p, envelope_csv="a.csv", margin=0.075, seed=1)
    assert h1 == h2 and h1 != h3
