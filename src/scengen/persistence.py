"""Persistence-tilted bootstrap for the Kirsch generator (prototype DU axis).

The Kirsch hybrid bootstrap draws its bootstrap index matrix ``M`` i.i.d. uniform per
(synthetic year, month) cell, so the synthetic interannual wet/dry sequence carries no
memory beyond the cross-year-boundary Cholesky correlation: multi-year drought
persistence is structurally absent from the unperturbed generator, and no adjustment of
the fitted moments (the mean/CV forcing axes) can create it.

This module injects interannual persistence THROUGH the bootstrap itself. A latent
annual wet/dry state ``w_t`` follows a standard-normal AR(1) with coefficient ``phi``;
each cell's bootstrap draw is tied to its year's state through a Gaussian copula with
loading ``lam`` and mapped to a historical year via the annual-wetness rank order:

    w_t   = phi * w_{t-1} + sqrt(1 - phi^2) * eta_t          eta_t  ~ N(0,1)
    g_tm  = sqrt(lam) * w_t + sqrt(1 - lam) * eps_tm         eps_tm ~ N(0,1)
    u_tm  = Phi(g_tm)                                        exactly U(0,1)
    M_tm  = wetness_order[ floor(u_tm * H) ]                 exactly uniform over years

Because ``g_tm`` is standard normal for every ``(phi, lam)``, each cell's marginal
stays EXACTLY uniform over the ``H`` historical years: the per-month bootstrap source
distribution — and with it the fitted monthly moments, the normal-score transform, and
the cross-site structure carried by the shared ``M`` — is preserved in distribution.
What changes is only the year-to-year DEPENDENCE of the drawn source years: adjacent
synthetic years preferentially resample wetness-similar historical years, which the
residual tensor converts into interannual autocorrelation of the output.

``lam = 0`` (or ``phi = 0``) reduces to i.i.d. uniform draws — the unperturbed Kirsch
generator, exactly. The price of ``lam > 0`` is a within-year coupling of the same
strength (all 12 cells of a year share ``w_t``), which inflates the within-year
month-to-month correlation above the fitted target; the accompanying diagnostic
(``NYCOptimization/scripts/supplemental/diagnose_persistence_axis.py``) measures that
distortion against the induced annual-lag-1 gain to decide whether the axis is
admissible.

Usage with a fitted :class:`synhydro.KirschGenerator`::

    order = wetness_rank_order(kirsch.Y)
    M = persistent_bootstrap_indices(
        n_years + 1, kirsch.n_periods_per_year, order, phi=phi, lam=lam, rng=rng,
    )
    df = kirsch.generate_single_series(n_years, M=M, as_array=False)

(the ``+ 1`` is the Kirsch cross-year buffer row).
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm


def wetness_rank_order(Y: np.ndarray) -> np.ndarray:
    """Order historical years by annual wetness of the standardized-residual tensor.

    Args:
        Y: Kirsch residual tensor of shape ``(n_hist_years, n_periods, n_sites)`` (the
            generator's ``.Y``, normal-scored when ``generate_using_log_flow``).

    Returns:
        Integer array ``order`` of length ``n_hist_years``: ``order[r]`` is the index of
        the year with wetness rank ``r`` (ascending, driest first), so uniform ranks map
        to uniform year indices.
    """
    Y = np.asarray(Y, dtype=float)
    if Y.ndim != 3:
        raise ValueError(f"Y must be (n_years, n_periods, n_sites); got shape {Y.shape}")
    wetness = Y.mean(axis=(1, 2))
    return np.argsort(wetness)


def persistent_bootstrap_indices(
    n_years: int,
    n_periods: int,
    wetness_order: np.ndarray,
    *,
    phi: float,
    lam: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw a persistence-tilted Kirsch bootstrap index matrix.

    Args:
        n_years: Number of index rows to draw (pass the Kirsch buffered length
            ``n_years + 1`` when feeding ``generate_single_series``).
        n_periods: Periods per year (12 monthly / 52 weekly).
        wetness_order: Ascending wetness rank -> year index map from
            :func:`wetness_rank_order`; its length is the number of historical years.
        phi: AR(1) coefficient of the latent annual state, in ``[0, 1)``.
        lam: Copula loading of each cell on its year's latent state, in ``[0, 1]``.
        rng: NumPy random generator (derive per realization from the SynHydro
            realization stream so draws stay keyed to the global index).

    Returns:
        Integer matrix ``M`` of shape ``(n_years, n_periods)`` with every cell marginally
        uniform over ``len(wetness_order)`` historical years.

    Raises:
        ValueError: If ``phi`` is outside ``[0, 1)`` or ``lam`` outside ``[0, 1]``.
    """
    if not 0.0 <= phi < 1.0:
        raise ValueError(f"phi must be in [0, 1); got {phi}")
    if not 0.0 <= lam <= 1.0:
        raise ValueError(f"lam must be in [0, 1]; got {lam}")
    order = np.asarray(wetness_order, dtype=int)
    n_hist = len(order)

    w = np.empty(n_years)
    w[0] = rng.standard_normal()
    innovations = rng.standard_normal(n_years - 1) if n_years > 1 else np.empty(0)
    scale = np.sqrt(1.0 - phi**2)
    for t in range(1, n_years):
        w[t] = phi * w[t - 1] + scale * innovations[t - 1]

    eps = rng.standard_normal((n_years, n_periods))
    g = np.sqrt(lam) * w[:, None] + np.sqrt(1.0 - lam) * eps
    ranks = np.minimum((norm.cdf(g) * n_hist).astype(int), n_hist - 1)
    return order[ranks]
