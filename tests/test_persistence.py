"""Tests for the persistence-tilted bootstrap (scengen.persistence)."""

import numpy as np
import pytest

from scengen.persistence import persistent_bootstrap_indices, wetness_rank_order


def _order(n_hist: int = 40) -> np.ndarray:
    rng = np.random.default_rng(7)
    Y = rng.standard_normal((n_hist, 12, 3))
    return wetness_rank_order(Y)


def test_wetness_rank_order_is_permutation():
    order = _order(40)
    assert sorted(order.tolist()) == list(range(40))


def test_wetness_rank_order_ascending():
    rng = np.random.default_rng(3)
    Y = rng.standard_normal((15, 12, 2))
    order = wetness_rank_order(Y)
    wetness = Y.mean(axis=(1, 2))
    assert np.all(np.diff(wetness[order]) >= 0)


@pytest.mark.parametrize("phi,lam", [(0.0, 0.0), (0.8, 0.0), (0.8, 0.5), (0.9, 1.0)])
def test_cell_marginal_is_uniform(phi, lam):
    """Every (phi, lam): per-cell marginal over historical years stays uniform.

    Cells within one matrix are dependent by design (that is the mechanism), so the
    uniformity check samples ONE fixed cell across many independent matrices - a
    genuinely i.i.d. sample of the cell marginal.
    """
    order = _order(20)
    rng = np.random.default_rng(11)
    draws = np.array([
        persistent_bootstrap_indices(3, 12, order, phi=phi, lam=lam, rng=rng)[1, 4]
        for _ in range(4000)
    ])
    counts = np.bincount(draws, minlength=20)
    expected = len(draws) / 20
    # chi-square 19 dof: p99 ~ 36; allow generous headroom for a smoke-level check
    chi2 = float(((counts - expected) ** 2 / expected).sum())
    assert chi2 < 60.0, f"cell marginal not uniform (chi2={chi2:.1f})"


def test_lam_zero_has_no_year_dependence():
    order = _order(30)
    rng = np.random.default_rng(5)
    M = persistent_bootstrap_indices(4000, 12, order, phi=0.9, lam=0.0, rng=rng)
    rank = np.argsort(order)[M]                     # back to wetness ranks
    year_mean = rank.mean(axis=1)
    r = np.corrcoef(year_mean[:-1], year_mean[1:])[0, 1]
    assert abs(r) < 0.05


def test_persistence_induces_year_dependence():
    order = _order(30)
    rng = np.random.default_rng(5)
    M = persistent_bootstrap_indices(4000, 12, order, phi=0.9, lam=0.5, rng=rng)
    rank = np.argsort(order)[M]
    year_mean = rank.mean(axis=1)
    r = np.corrcoef(year_mean[:-1], year_mean[1:])[0, 1]
    assert r > 0.3


def test_shape_and_bounds():
    order = _order(25)
    rng = np.random.default_rng(0)
    M = persistent_bootstrap_indices(11, 12, order, phi=0.5, lam=0.3, rng=rng)
    assert M.shape == (11, 12)
    assert M.min() >= 0 and M.max() < 25


@pytest.mark.parametrize("kwargs", [{"phi": 1.0, "lam": 0.5}, {"phi": -0.1, "lam": 0.5},
                                    {"phi": 0.5, "lam": 1.1}, {"phi": 0.5, "lam": -0.1}])
def test_invalid_params_raise(kwargs):
    order = _order(10)
    with pytest.raises(ValueError):
        persistent_bootstrap_indices(5, 12, order, rng=np.random.default_rng(0), **kwargs)
