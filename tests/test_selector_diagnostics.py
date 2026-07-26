"""Tests for the selector-comparison diagnostics (pure numpy/scipy; no SSI)."""

from __future__ import annotations

import numpy as np
import pytest

from scengen import selector_diagnostics as sd
from scengen import subsample as ss


def _pool(M=400, seed=0):
    """Skewed 3-axis pool with a dry-style zero atom on axis 0."""
    rng = np.random.default_rng(seed)
    H = rng.gamma(2.0, 1.0, size=(M, 3))
    # 20% zero-event windows: ALL dry descriptors are zero when no event occurs.
    H[: M // 5, :2] = 0.0
    return H, ["drought_deficit_volume", "drought_onset_rate", "flood_peak_magnitude"]


@pytest.mark.parametrize("name", sorted(sd.SELECTORS))
def test_each_selector_returns_valid_subset(name):
    H, axes = _pool()
    X = ss.minmax_normalize(H)
    res = sd.SELECTORS[name](X, 30, seed=1)
    assert res.rows.shape == (30,)
    assert len(set(res.rows.tolist())) == 30
    assert np.all(np.diff(res.rows) > 0)
    assert set(res.rows.tolist()) <= set(range(len(H)))


@pytest.mark.parametrize("name", sorted(sd.SELECTORS))
def test_each_selector_is_deterministic_given_seed(name):
    H, _ = _pool()
    X = ss.minmax_normalize(H)
    a = sd.SELECTORS[name](X, 25, seed=7).rows
    b = sd.SELECTORS[name](X, 25, seed=7).rows
    np.testing.assert_array_equal(a, b)


def test_lhs_selectors_report_snap_distances():
    H, _ = _pool()
    X = ss.minmax_normalize(H)
    for name in ("lhs_nn", "lhs_assign"):
        info = sd.SELECTORS[name](X, 20, seed=0).info
        assert info["snap_distances"].shape == (20,)
        assert np.all(info["snap_distances"] >= 0.0)


def test_eps_cell_guarantees_one_member_per_cell():
    H, _ = _pool()
    X = ss.minmax_normalize(H)
    res = sd.select_eps_cell(X, 40, seed=3)
    g = res.info["grid_resolution"]
    assert res.info["n_occupied_cells"] >= 40
    cells = np.minimum((X[res.rows] * g).astype(int), g - 1)
    assert len({tuple(c) for c in cells}) == 40  # no two share a cell


def test_eps_cell_never_selects_off_manifold():
    """Trivially true (it selects pool members), but the calibration must be minimal:
    resolution g-1 must NOT reach n occupied cells."""
    H, _ = _pool(M=300)
    X = ss.minmax_normalize(H)
    res = sd.select_eps_cell(X, 50, seed=0)
    g = res.info["grid_resolution"]
    if g > 1:
        coarser = np.minimum((X * (g - 1)).astype(int), g - 2)
        assert len({tuple(c) for c in coarser}) < 50


def test_selection_metrics_battery_keys_and_atom():
    H, axes = _pool()
    rows = ss.random_subsample(H, 30, seed=0)
    m = sd.selection_metrics(H, rows, axes)
    for key in (
        "L2_star_abs", "L2_star_cdf", "nn_min_abs", "tail_share_p90",
        "corner_share_p90", "ks_mean_vs_pool", "mst_edge_mean",
        "zero_event_share_selected", "zero_event_share_pool",
    ):
        assert key in m and np.isfinite(m[key])
    assert m["zero_event_share_pool"] == pytest.approx(0.2, abs=0.01)


def test_run_selector_comparison_table_shape_and_stability():
    H, axes = _pool()
    table, details = sd.run_selector_comparison(
        H, axes, 24, seeds=[0, 1, 2], selectors=("random", "lhs_nn", "maximin"),
    )
    assert len(table) == 9  # 3 selectors x 3 seeds
    assert set(table["selector"]) == {"random", "lhs_nn", "maximin"}
    # maximin is deterministic: identical rows across seeds -> Jaccard 1.
    assert details["maximin"]["jaccard_across_seeds"] == pytest.approx(1.0)
    assert 0.0 <= details["random"]["jaccard_across_seeds"] < 1.0


def test_designed_selectors_beat_random_on_abs_coverage():
    """Sanity: every designed rule covers abs space better than random on average."""
    H, axes = _pool(M=500, seed=4)
    table, _ = sd.run_selector_comparison(
        H, axes, 30, seeds=[0, 1, 2, 3],
        selectors=("random", "lhs_nn", "lhs_assign", "maximin", "eps_cell"),
    )
    mean_l2 = table.groupby("selector")["L2_star_abs"].mean()
    for name in ("lhs_nn", "lhs_assign", "eps_cell"):
        assert mean_l2[name] < mean_l2["random"], name
