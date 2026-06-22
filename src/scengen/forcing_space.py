"""Forcing space: hybrid CMIP6 envelope + LHS fill (methods 3.1).

Builds the deeply-uncertain forcing-parameter space ``theta`` (12-month multiplicative change-factor
profiles) and applies it externally to a fitted Kirsch generator via the log-space transform
(Kirsch et al. 2013, eqs. 10-11). The same forcing space drives the deeply-uncertain / resampled
generation and the test ensemble.

CMIP6 source: the *multiplicative* ``_frac_`` twin CSV (already a_j; baseline column == 1.0)::

    ../CMIP6_multimodel_streamflow/stats/diff_relative_to_dataset_baseline/
        nyc_inflow_monthly_mean_frac_by_dataset_ssp_and_period.csv

The CSV month index is CALENDAR (1=Jan); Kirsch/Pywr-DRB are water-year aligned (Oct start) -> the
profiles MUST be re-indexed calendar->water-year before they are applied to ``mean_period``/``std_period``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd

#: Default location of the multiplicative CMIP6 monthly change-factor table (a_j), relative to the
#: sibling CMIP6 repo. Resolve against the actual checkout at call time.
CMIP6_FRAC_CSV: Final[str] = (
    "CMIP6_multimodel_streamflow/stats/diff_relative_to_dataset_baseline/"
    "nyc_inflow_monthly_mean_frac_by_dataset_ssp_and_period.csv"
)

#: Water-year month order (Oct..Sep) used by the Kirsch ``period`` index.
WATER_YEAR_MONTHS: Final[tuple[int, ...]] = (10, 11, 12, 1, 2, 3, 4, 5, 6, 7, 8, 9)


def load_cmip6_envelope(csv_path: str | Path) -> pd.DataFrame:
    """Load the multiplicative CMIP6 change-factor profiles.

    Args:
        csv_path: Path to the ``_frac_`` CSV (12 calendar-month rows x scenario columns).

    Returns:
        DataFrame of multiplicative factors ``a_j`` indexed by water-year month order
        (Oct..Sep), one column per CMIP6 scenario. Confirm the file is the ``_frac_`` variant
        (values near 1.0), not the ``_prc_change_`` percent file.

    Raises:
        NotImplementedError: stub.
    """
    raise NotImplementedError("methods 3.1: load _frac_ CSV; re-index calendar->water-year")


def calendar_to_water_year(profile: np.ndarray) -> np.ndarray:
    """Reorder a 12-vector from calendar-month (Jan..Dec) to water-year (Oct..Sep) order."""
    if profile.shape[-1] != 12:
        raise ValueError("profile last axis must be length 12 (months)")
    idx = [m - 1 for m in WATER_YEAR_MONTHS]
    return profile[..., idx]


def sample_forcing_space(
    n_profiles: int,
    envelope: pd.DataFrame,
    *,
    seed: int,
    margin: float = 0.075,
    pc_rotated: bool = True,
) -> np.ndarray:
    """Draw ``theta`` profiles by LHS over the CMIP6 envelope plus a modest exterior margin.

    Args:
        n_profiles: Number of forcing profiles to draw (N_Theta).
        envelope: Water-year-ordered CMIP6 anchor profiles (output of :func:`load_cmip6_envelope`).
        seed: RNG seed.
        margin: Fractional extrapolation beyond the per-month [min, max] envelope (delta).
        pc_rotated: If True, draw the 12-D LHS in the principal-component basis of the anchor
            profiles and rotate back, preserving seasonal co-variation; if False, independent-month LHS.

    Returns:
        Array of shape ``(n_profiles, 12)`` of multiplicative monthly factors ``a_j`` in water-year order.

    Raises:
        NotImplementedError: stub. Reuse ``MOEA-FIND/src/discovery/analysis.py::generate_lhs_samples``.
    """
    raise NotImplementedError("methods 3.1: PC-rotated LHS over [min-margin, max+margin]")


def apply_climate_adjustment(
    mean_period: np.ndarray,
    std_period: np.ndarray,
    a_profile: np.ndarray,
    *,
    preserve_absolute_sd: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    r"""Apply a multiplicative monthly change profile to log-space Kirsch statistics.

    Implements Kirsch et al. (2013) eqs. 10-11 (the transform used in
    ``StochasticExploratoryExperiment/methods/generate.py``). With baseline log-space
    ``(Y_bar_j, sigma_j)``, factor ``a_j``, and ``c_j = 1`` (preserve absolute real-space SD):

        Y_bar_new = ln(a) + Y_bar + sigma^2/2 - 0.5 * ln( a^-2 (exp(sigma^2) - 1) + 1 )
        sigma_new = sqrt( ln( a^-2 (exp(sigma^2) - 1) + 1 ) )

    Args:
        mean_period: Baseline log-space per-month means ``Y_bar_j`` (water-year ordered).
        std_period: Baseline log-space per-month SDs ``sigma_j``.
        a_profile: Multiplicative monthly factors ``a_j`` (water-year ordered), from
            :func:`sample_forcing_space`.
        preserve_absolute_sd: If True use ``c_j = 1`` (preserve absolute real-space SD; CV not
            preserved). The CV-preserving variant (``c_j = a_j``) is a flagged alternative.

    Returns:
        ``(mean_period_new, std_period_new)`` to assign onto the fitted generator before generation.
    """
    if not preserve_absolute_sd:
        raise NotImplementedError("CV-preserving variant (c_j = a_j) not yet implemented")
    a = np.asarray(a_profile, dtype=float)
    y_bar = np.asarray(mean_period, dtype=float)
    sigma = np.asarray(std_period, dtype=float)
    arg = (1.0 / a**2) * (np.exp(sigma**2) - 1.0) + 1.0
    y_bar_new = np.log(a) + y_bar + sigma**2 / 2.0 - 0.5 * np.log(arg)
    sigma_new = np.sqrt(np.log(arg))
    return y_bar_new, sigma_new


def forcing_hash(profiles: np.ndarray, *, envelope_csv: str | Path, margin: float, seed: int) -> str:
    """Return a stable hash of the forcing-space configuration for the provenance manifest.

    Uses a content hash (e.g. ``hashlib.sha1``) over the profiles + config, NOT the process-salted
    builtin ``hash`` (determinism requirement).

    Raises:
        NotImplementedError: stub.
    """
    raise NotImplementedError("hashlib-based stable hash of (profiles, csv, margin, seed)")
