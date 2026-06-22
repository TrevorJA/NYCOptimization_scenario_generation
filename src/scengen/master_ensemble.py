"""Master-ensemble generation + streaming hazard image (methods 3.2).

Generates the large master ensemble M (~1e6 short realizations) with the Kirsch-Nowak pipeline under
the hybrid forcing space, and computes its hazard image H by streaming (daily traces are discarded as
H is accumulated). Daily traces are NOT stored; only H + generator parameters + per-realization seeds
are persisted, and selected realizations are regenerated on demand.

Determinism (provided upstream by SynHydro commit 7659704 ``synhydro.core.seeding``): realization k is
fully determined by a child stream keyed to the GLOBAL index k, invariant to MPI layout / batch
partition. Use the SynHydro public API rather than reimplementing seeding:

    KirschGenerator.generate(..., realization_indices=[k0, k1, ...], seed=master_seed)  # keyed by global idx
    NowakDisaggregator.disaggregate(monthly_ensemble, seed=master_seed)                 # global idx = int key
    KirschNowakPipeline.generate(..., realization_indices=..., seed=master_seed)         # forwards to both

Two Kirsch generators are fit: one on the baseline period (supplying mean_period/std_period for the
eqs-10-11 climate adjustment) and one on the full record.

Ports generation/staging logic from ``../NYCOptimization/src/ensemble_generation.py`` (which has the
climate-adjustment branch stripped, and -- independent of the now-fixed SynHydro RNG -- a process-salted
``hash(kde_name)`` and a batched KDE fill that couples realizations). The port must: re-add the climate
branch; replace ``hash(kde_name)`` with ``zlib.crc32``; do the KDE downstream-node fill PER REALIZATION
(``kde.resample(n_days, ...)`` seeded from realization k's child stream, namespaced by
``crc32(pair_name)``); and PRESERVE the integer realization keys across generate->disaggregate (the Nowak
sub-stream is keyed to those integer keys -- re-keying changes the daily output).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from .manifest import EnsembleManifest


@dataclass(frozen=True)
class MasterEnsembleConfig:
    """Configuration for one master-ensemble generation.

    Attributes:
        master_seed: Root seed; per-realization streams are ``SeedSequence(master_seed).spawn(N)``.
        n_forcing_profiles: Number of theta profiles (N_Theta).
        realizations_per_profile: Realizations generated per forcing profile (n_per_theta).
        realization_years: Disjoint window length L (years); must exceed the longest within-window
            drought (check vs the 1960s DRB drought of record ~4-5 yr).
        flowtype: Historical full-record flow dataset key (e.g. ``pub_nhmv10_BC_withObsScaled``).
        baseline_period: (start, end) for the baseline-fit Kirsch (eqs 10-11 reference).
        full_period: (start, end) for the full-record-fit Kirsch.
        output_dir: Staging directory (under STAGED_ENSEMBLE_DIR); H + seeds + params land here.
        hazard_axes: Screened hazard-metric axis names to compute for H (from :mod:`hazard_metrics`).
    """

    master_seed: int
    n_forcing_profiles: int
    realizations_per_profile: int
    realization_years: int
    flowtype: str = "pub_nhmv10_BC_withObsScaled"
    baseline_period: tuple[str, str] = ("1980-10-01", "2019-09-30")
    full_period: tuple[str, str] = ("1945-10-01", "2022-09-30")
    output_dir: Path | None = None
    hazard_axes: tuple[str, ...] = ()
    extra: Mapping[str, Any] = field(default_factory=dict)

    @property
    def n_realizations(self) -> int:
        """Total master-ensemble cardinality N_M."""
        return self.n_forcing_profiles * self.realizations_per_profile


def child_seed_sequence(master_seed: int, n: int) -> list[np.random.SeedSequence]:
    """Return the ``n`` per-realization child seed sequences keyed by GLOBAL index.

    ``children[k]`` is the stream for global realization ``k`` and is invariant to how realizations
    are partitioned across MPI ranks/batches. This mirrors the canonical SynHydro helpers in
    ``synhydro.core.seeding`` (``spawn_realization_seed`` / ``realization_rng``); prefer those when
    generating, and especially ``spawn_realization_seed(master, k)`` for isolated regeneration (it
    reconstructs ``spawn(N)[k]`` from the spawn key without materializing all ``n`` children).
    """
    return list(np.random.SeedSequence(master_seed).spawn(n))


def generate_master_ensemble(config: MasterEnsembleConfig) -> EnsembleManifest:
    """Generate M, stream its hazard image H, and persist H + seeds + params + manifest.

    Steps (methods 3.2):
        1. Fit baseline-period and full-record Kirsch generators; fit Nowak.
        2. For each forcing profile theta_i: derive (mean_period_new, std_period_new) via
           ``forcing_space.apply_climate_adjustment`` and generate ``realizations_per_profile``
           realizations, each driven by its global-index child RNG.
        3. For each realization: Nowak daily disaggregation + per-realization KDE downstream fill +
           ``_subtract_upstream_catchment_inflows``; compute h(.) (methods 3.3); discard daily traces.
        4. Persist H (N_M x m), generator params, per-realization seeds; write the manifest.

    Returns:
        The :class:`EnsembleManifest` describing M (design ``"master"``).

    Raises:
        NotImplementedError: stub.
    """
    raise NotImplementedError("methods 3.2: streaming generation + hazard-image accumulation")


def regenerate_realization(
    master_seed: int,
    global_index: int,
    *,
    config: MasterEnsembleConfig,
) -> pd.DataFrame:
    """Regenerate a single master-ensemble realization bit-for-bit from its global index.

    Uses the deterministic SynHydro per-realization-RNG API:
    ``KirschGenerator.generate(realization_indices=[global_index], seed=master_seed)`` then
    ``NowakDisaggregator.disaggregate(monthly, seed=master_seed)`` on the key-``global_index`` ensemble,
    then the per-realization KDE downstream fill -- identical regardless of N or MPI layout. Validated
    by an end-to-end determinism regression test (generate the realization under two partitions; assert
    array-equal through the KDE fill, post float32/HDF5).

    Returns:
        Daily multi-node flows for the requested realization.

    Raises:
        NotImplementedError: stub.
    """
    raise NotImplementedError("methods 3.2: deterministic single-realization regeneration")
