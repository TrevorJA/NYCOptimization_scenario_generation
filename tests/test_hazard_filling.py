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
    # Keep-all policy: every non-degenerate axis retained minus near-duplicates.
    assert 3 <= len(out["chosen_axes"]) <= 8
    assert set(out["chosen_axes"]) <= set(hm.CANDIDATE_EVENT_METRICS)
    assert out["chosen_axes"] == out["screen"]["retained"]
    # No retained pair may exceed the near-duplicate threshold.
    kept_idx = [out["screen"]["spearman_axes"].index(a) for a in out["chosen_axes"]
                if a in out["screen"]["spearman_axes"]]
    rho = np.asarray(out["screen"]["spearman_rho"])
    for i, a in enumerate(kept_idx):
        for b in kept_idx[i + 1:]:
            assert abs(rho[a, b]) < hf.DEDUPE_RHO_THRESHOLD


def test_select_from_candidate_image_selection_axes_restriction():
    """A caller-fixed selection axis set restricts the screen and the snap to it.

    The policy hook for a campaign axis set chosen by a pool-size saturation
    diagnostic: the other image columns stay computed but never enter selection.
    """
    H, axes = _candidate_image(n_scen=60)
    subset = [a for a in axes if a not in ("drought_duration", "flood_rise_rate")]
    out = hf.select_from_candidate_image(
        H, axes, n=10, seed=0, selector_space="abs", selection_axes=subset,
    )
    assert out["candidate_axes"] == subset
    assert set(out["chosen_axes"]) <= set(subset)
    assert out["H_candidates"].shape == (60, len(subset))
    assert out["selected_rows"].shape == (10,)

    with pytest.raises(ValueError, match="selection_axes"):
        hf.select_from_candidate_image(
            H, axes, n=10, seed=0, selector_space="abs",
            selection_axes=["not_a_real_axis"],
        )


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
    # Face-resident share of SELECTED members is reported (tail-seeking selector
    # picks clipped members more often than their pool share).
    assert 0.0 <= norm["selected_face_resident_frac"] <= 1.0
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


def _screen_pool(M=500, seed=0):
    """Five independent axes + one duplicate of axis 0 + one constant axis."""
    rng = np.random.default_rng(seed)
    base = rng.gamma(2.0, 1.0, size=(M, 5))
    dup = base[:, [0]] * 3.0 + 0.01 * rng.normal(size=(M, 1))  # rank-duplicate of ax0
    const = np.full((M, 1), 7.0)
    H = np.hstack([base, dup, const])
    names = ["ax0", "ax1", "ax2", "ax3", "ax4", "ax0_dup", "ax_const"]
    return H, names


def test_screen_keeps_all_non_degenerate_non_duplicate_axes():
    H, names = _screen_pool()
    screen = hf.screen_hazard_axes(H, names, axis_priority=("ax0",))
    assert screen["retained"] == ["ax0", "ax1", "ax2", "ax3", "ax4"]
    assert screen["dropped"]["ax_const"]["reason"] == "degenerate"
    assert screen["dropped"]["ax0_dup"]["reason"] == "near_duplicate"
    assert screen["dropped"]["ax0_dup"]["kept_member"] == "ax0"
    assert abs(screen["dropped"]["ax0_dup"]["rho_with_kept"]) >= hf.DEDUPE_RHO_THRESHOLD
    assert ["ax0", "ax0_dup"] in screen["near_duplicate_groups"]


def test_screen_priority_picks_the_canonical_group_member():
    H, names = _screen_pool()
    screen = hf.screen_hazard_axes(H, names, axis_priority=("ax0_dup", "ax0"))
    assert "ax0_dup" in screen["retained"] and "ax0" not in screen["retained"]
    assert screen["dropped"]["ax0"]["kept_member"] == "ax0_dup"


def test_screen_threshold_is_configurable():
    """Tightening the threshold prunes correlated-but-distinct axes too."""
    rng = np.random.default_rng(1)
    x = rng.gamma(2.0, 1.0, size=(800, 1))
    corr = x + 0.35 * rng.normal(size=(800, 1)) * x.std()  # |rho_S| ~ 0.9, < 0.95
    indep = rng.gamma(2.0, 1.0, size=(800, 2))
    H = np.hstack([x, corr, indep])
    names = ["ax0", "ax0_corr", "ax1", "ax2"]
    loose = hf.screen_hazard_axes(H, names, dedupe_threshold=0.95)
    tight = hf.screen_hazard_axes(H, names, dedupe_threshold=0.80,
                                  axis_priority=("ax0",))
    assert loose["retained"] == names  # correlated pair below 0.95 both kept
    assert tight["retained"] == ["ax0", "ax1", "ax2"]


def test_screen_persists_full_spearman_matrix_json_safe():
    H, names = _screen_pool()
    screen = hf.screen_hazard_axes(H, names)
    kept = screen["spearman_axes"]
    assert "ax_const" not in kept  # matrix spans the non-degenerate axes only
    rho = np.asarray(screen["spearman_rho"])
    assert rho.shape == (len(kept), len(kept))
    np.testing.assert_allclose(np.diag(rho), 1.0)
    json.loads(json.dumps(screen))


def test_screen_rejects_fewer_than_three_retained_axes():
    """A pathologically degenerate/redundant pool must raise, not pass m=2."""
    rng = np.random.default_rng(2)
    a = rng.gamma(2.0, 1.0, size=(400, 1))
    H = np.hstack([a, a * 2.0, np.full((400, 1), 1.0)])
    with pytest.raises(ValueError, match="at least 3"):
        hf.screen_hazard_axes(H, ["ax0", "ax0_dup", "ax_const"])


def test_absolute_selector_beats_null_in_its_own_geometry():
    """Method verification: the campaign selector fills absolute space better than random."""
    H, axes = _candidate_image(n_scen=200)
    cov = hf.select_from_candidate_image(
        H, axes, n=24, seed=0, selector_space="abs",
    )["coverage"]["geometries"]["abs"]
    assert cov["selected_L2_star"] < cov["null_mean"]
    assert cov["percentile"] < 50.0
