"""Tests for the coverage / build-QC diagnostics (pure numpy/scipy)."""

from __future__ import annotations

import numpy as np

from scengen import diagnostics as dg
from scengen import subsample as ss


def _clustered_hazard_image(M=300, d=3, seed=0):
    rng = np.random.default_rng(seed)
    n_tail = max(1, M // 10)
    blob = rng.normal(0.0, 1.0, size=(M - n_tail, d))
    tail = rng.normal(5.0, 2.0, size=(n_tail, d))
    return np.vstack([blob, tail])


def test_coverage_report_prefers_space_filling_over_random():
    H = _clustered_hazard_image(seed=1)
    n = 24
    sel = ss.hazard_filling_subsample(H, n, seed=0)
    rnd = ss.random_subsample(H, n, seed=0)

    rep_sel = dg.coverage_report(H, sel, n_boot=200, seed=0)
    rep_rnd = dg.coverage_report(H, rnd, n_boot=200, seed=0)

    assert rep_sel.n == n and rep_sel.m == 3
    # The LHS+NN subset is more uniform (lower discrepancy, lower percentile).
    assert rep_sel.l2_star < rep_rnd.l2_star
    assert rep_sel.discrepancy_percentile <= rep_rnd.discrepancy_percentile
    assert 0.0 <= rep_sel.discrepancy_percentile <= 1.0


def test_marginal_gaps_shape_and_bounds():
    H = _clustered_hazard_image()
    sel = ss.hazard_filling_subsample(H, 20, seed=2)
    gaps = dg.marginal_gaps(H, sel)
    assert gaps.shape == (3,)
    assert np.all(gaps > 0) and np.all(gaps <= 1.0)


def test_redundancy_screen_flags_correlated_axes():
    rng = np.random.default_rng(0)
    a = rng.normal(size=400)
    H = np.column_stack([a, a + 0.01 * rng.normal(size=400), rng.normal(size=400)])
    out = dg.redundancy_screen(H, ["x", "x_copy", "indep"], threshold=0.7)
    flagged = {tuple(sorted((i, j))) for i, j, _ in out["redundant_pairs"]}
    assert ("x", "x_copy") in flagged
    assert ("indep", "x") not in flagged


def test_save_load_hazard_image_roundtrip(tmp_path):
    H = _clustered_hazard_image(M=50, d=3)
    sel = ss.hazard_filling_subsample(H, 8, seed=0)
    path = dg.save_hazard_image(
        tmp_path / "hazard_image.npz",
        H=H, hazard_axes=["a", "b", "c"],
        realization_ids=list(range(50)), selected_rows=sel,
    )
    back = dg.load_hazard_image(path)
    np.testing.assert_allclose(back["H"], H)
    assert back["hazard_axes"] == ["a", "b", "c"]
    np.testing.assert_array_equal(back["selected_rows"], sel)
