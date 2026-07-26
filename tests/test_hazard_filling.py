"""Tests for the hazard-filling driver. Skipped when SynHydro is unavailable."""

from __future__ import annotations

import json

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
    """The live entry point: screen a precomputed candidate image, then select.

    Run in the CAMPAIGN selector space (absolute, range-scaled magnitude).
    """
    H, axes = _candidate_image(n_scen=60)
    assert axes == list(hm.CANDIDATE_EVENT_METRICS)
    assert H.shape == (60, 8)

    out = hf.select_from_candidate_image(H, axes, n=10, seed=0, selector_space="abs")

    assert out["selected_rows"].shape == (10,)
    assert set(out["selected_rows"].tolist()) <= set(range(60))
    assert out["candidate_axes"] == list(hm.CANDIDATE_EVENT_METRICS)
    assert out["H_candidates"].shape == (60, 8)
    # tail-balanced chosen set: <=2 dry + <=2 wet (<=4 total), subset of candidates.
    assert 1 <= len(out["chosen_axes"]) <= 4
    assert set(out["chosen_axes"]) <= set(hm.CANDIDATE_EVENT_METRICS)
    assert sum(a.startswith("drought") for a in out["chosen_axes"]) <= 2
    assert sum(a.startswith("flood") for a in out["chosen_axes"]) <= 2


def test_select_from_candidate_image_is_deterministic_given_seed():
    """LHS + NN-snap has no stochastic search: the same seed must reproduce the design."""
    H, axes = _candidate_image(n_scen=60)
    kw = dict(n=10, seed=3, selector_space="abs")
    a = hf.select_from_candidate_image(H, axes, **kw)["selected_rows"]
    b = hf.select_from_candidate_image(H, axes, **kw)["selected_rows"]
    np.testing.assert_array_equal(a, b)


def test_select_from_candidate_image_cdf_space():
    """The retained non-campaign sensitivity: selection in empirical-CDF/rank space."""
    H, axes = _candidate_image(n_scen=60)
    out = hf.select_from_candidate_image(H, axes, n=10, seed=0, selector_space="cdf")
    assert out["selected_rows"].shape == (10,)


def test_select_from_candidate_image_rejects_unknown_space():
    H, axes = _candidate_image(n_scen=60)
    with pytest.raises(ValueError):
        hf.select_from_candidate_image(H, axes, n=10, seed=0, selector_space="nope")


def test_selector_space_is_required():
    """Omitting it would silently pick an arm, so it must be an explicit choice."""
    H, axes = _candidate_image(n_scen=60)
    with pytest.raises(TypeError):
        hf.select_from_candidate_image(H, axes, n=10, seed=0)


def test_coverage_qc_reports_both_geometries_against_a_null():
    """Coverage QC: L2-star in both geometries, placed inside a random-subset null."""
    H, axes = _candidate_image(n_scen=60)
    out = hf.select_from_candidate_image(
        H, axes, n=10, seed=1, selector_space="abs",
    )
    cov = out["coverage"]
    assert cov["n_selected"] == 10 and cov["n_pool"] == 60 and cov["n_null"] > 1
    assert set(cov["geometries"]) == {"cdf", "abs"}
    for geom in ("cdf", "abs"):
        c = cov["geometries"][geom]
        assert set(c) == {
            "selected_L2_star", "null_mean", "null_std",
            "null_min", "null_max", "percentile",
        }
        assert all(isinstance(v, float) for v in c.values())
        assert c["null_min"] <= c["null_mean"] <= c["null_max"]
        assert 0.0 <= c["percentile"] <= 100.0
    # JSON-serializable: it is persisted verbatim into the staged _meta.json.
    json.loads(json.dumps(cov))


def test_coverage_qc_null_is_deterministic_given_seed():
    """The null draws are derived from the design seed, so QC is reproducible."""
    H, axes = _candidate_image(n_scen=60)
    kw = dict(n=10, seed=2, selector_space="abs")
    a = hf.select_from_candidate_image(H, axes, **kw)["coverage"]
    b = hf.select_from_candidate_image(H, axes, **kw)["coverage"]
    assert a == b


def test_normalization_qc_reports_robust_bounds_and_clipped_mass():
    """The abs-space bounds and per-axis clipped fractions are persisted build-QC."""
    H, axes = _candidate_image(n_scen=200)
    out = hf.select_from_candidate_image(H, axes, n=24, seed=0, selector_space="abs")
    norm = out["normalization"]
    assert norm["lo_pct"] == 1.0 and norm["hi_pct"] == 99.0  # campaign p1/p99
    assert set(norm["axes"]) == set(out["chosen_axes"])
    for qc in norm["axes"].values():
        assert qc["hi"] > qc["lo"]
        # Clipped mass per side is bounded by the percentile it was cut at.
        assert 0.0 <= qc["clipped_low_frac"] <= 0.02
        assert 0.0 <= qc["clipped_high_frac"] <= 0.02
    json.loads(json.dumps(norm))


def test_selector_kwargs_bounds_propagate_to_qc():
    """Overriding lo/hi_pct must move the selector, coverage QC, and normalization QC together."""
    H, axes = _candidate_image(n_scen=100)
    out = hf.select_from_candidate_image(
        H, axes, n=10, seed=0, selector_space="abs",
        selector_kwargs={"lo_pct": 0.0, "hi_pct": 100.0},
    )
    norm = out["normalization"]
    assert norm["lo_pct"] == 0.0 and norm["hi_pct"] == 100.0
    for qc in norm["axes"].values():
        assert qc["clipped_low_frac"] == 0.0 and qc["clipped_high_frac"] == 0.0


def test_absolute_selector_beats_null_in_its_own_geometry():
    """Method verification: the campaign selector fills absolute space better than random."""
    H, axes = _candidate_image(n_scen=200)
    cov = hf.select_from_candidate_image(
        H, axes, n=24, seed=0, selector_space="abs",
    )["coverage"]["geometries"]["abs"]
    assert cov["selected_L2_star"] < cov["null_mean"]
    assert cov["percentile"] < 50.0
