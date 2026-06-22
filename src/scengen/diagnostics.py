"""Ensemble-quality / build-QC diagnostics (methods 6a).

These are *build-QC gates* that verify an ensemble was constructed as specified -- they are NOT
evidence of an outcome advantage (the overfitting-gap and stability hypotheses in methods 6b live in
``../NYCOptimization`` because they need MOEA re-evaluation results). In particular, L2-star
discrepancy IS the hazard-filling objective, so it is a build-verification gate, never proof of value.

Reuses ``coverage_metrics`` from ``../MOEA-FIND/src/discovery/analysis.py`` (L2-star discrepancy,
nearest-neighbor mean/std/min/max, CV). Statistical fidelity is a within-faithful-arm check only;
``hazard_filling`` is exempt and never ranked on fidelity.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class CoverageReport:
    """Coverage diagnostics for one ensemble's hazard coordinates.

    Attributes:
        l2_star_discrepancy: L2-star discrepancy (lower = more uniform). QC gate.
        nn_min: Minimum nearest-neighbor distance (separation).
        nn_cv: Coefficient of variation of nearest-neighbor distances (lower = more uniform).
        mst_mean: Mean minimum-spanning-tree edge length.
        effective_sample_size: Theta-discounted effective sample size.
        expected_random_discrepancy: Expected L2-star of a random design at the same (N, m), for
            coverage-in-context (so a coverage advantage is reported relative to chance, not asserted).
    """

    l2_star_discrepancy: float
    nn_min: float
    nn_cv: float
    mst_mean: float
    effective_sample_size: float
    expected_random_discrepancy: float


def coverage_report(H_sub: np.ndarray, *, n_theta: int | None = None) -> CoverageReport:
    """Compute the coverage QC diagnostics for a subsampled ensemble's hazard coordinates.

    Args:
        H_sub: Selected ensemble's hazard coordinates (N_d x m), normalized per axis.
        n_theta: Number of distinct forcing draws represented (for the theta-discounted ESS).

    Raises:
        NotImplementedError: stub. Wrap ``moeafind ... analysis.coverage_metrics``.
    """
    raise NotImplementedError("methods 6a: wrap MOEA-FIND coverage_metrics + ESS + context")


def expected_random_discrepancy(n: int, m: int, *, n_boot: int = 200, seed: int = 0) -> float:
    """Bootstrap the expected L2-star discrepancy of a random design at given (N, m).

    Raises:
        NotImplementedError: stub.
    """
    raise NotImplementedError("methods 6a: bootstrap random-design discrepancy")


def redundancy_clusters(H_sub: np.ndarray, metric_names: Sequence[str]) -> Mapping[int, Sequence[str]]:
    """Report the distinct redundancy clusters represented in a subsampled ensemble (QC).

    Pass condition: spans >= m clusters (not collapsed onto a redundant subspace).

    Raises:
        NotImplementedError: stub.
    """
    raise NotImplementedError("methods 6a: redundancy-cluster coverage")


@dataclass(frozen=True)
class FidelityReport:
    """Within-faithful-arm fidelity of an ensemble to the parent record/master ensemble.

    Attributes:
        monthly_mean_bias: Per-month mean-flow bias vs Q_obs / M.
        monthly_sd_bias: Per-month SD bias.
        lag1_autocorr_match: Lag-1 autocorrelation agreement.
        cross_site_corr_match: Cross-site correlation-matrix agreement.
        fdc_band_ok: Whether the flow-duration curve falls within tolerance.
        drought_sdf_ok: Whether the drought severity-duration-frequency surface brackets history.
    """

    monthly_mean_bias: Sequence[float] = field(default_factory=list)
    monthly_sd_bias: Sequence[float] = field(default_factory=list)
    lag1_autocorr_match: float = float("nan")
    cross_site_corr_match: float = float("nan")
    fdc_band_ok: bool = False
    drought_sdf_ok: bool = False


def statistical_fidelity(ensemble, q_obs) -> FidelityReport:  # noqa: ANN001 (stub signature)
    """Generator-validity check for FAITHFUL-arm designs only (methods 6a).

    ``hazard_filling`` (and any distorted-arm design) is exempt and must NOT be ranked against
    faithful designs on fidelity -- it distorts marginals by construction. Use only to confirm each
    selected scenario is itself a valid generator output.

    Raises:
        NotImplementedError: stub.
    """
    raise NotImplementedError("methods 6a: faithful-arm fidelity to Q_obs / M")
