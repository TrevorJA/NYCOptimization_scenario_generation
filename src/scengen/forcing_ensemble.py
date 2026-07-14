"""Forcing-ensemble generation + optional streaming hazard image (methods 3.2).

Generates a Kirsch-Nowak ensemble of short realizations under one of two populations:

    ``stationary``  Kirsch-Nowak fit to the historic record, NO climate perturbation. The generator
                    keeps the moments it was fitted with on the full record.
    ``du_forced``   forcing parameters ``theta`` drawn from the CMIP6 harmonic hypercube; each
                    profile's Kirsch moments are climate-adjusted (Kirsch et al. 2013 eqs. 10-11)
                    against a baseline-period reference generator.

Every scenario design generates its own realizations from its own namespaced seed stream
(:mod:`scengen.seeds`); there is no shared ensemble across designs. A design that later subsamples
(hazard filling) owns a **candidate pool**, which must be sampled i.i.d. (``theta_sampler="iid"``)
for the cross-design distributional-equivalence control to hold — see
:func:`scengen.forcing_space.sample_harmonic_forcing`.

Determinism (provided upstream by SynHydro ``synhydro.core.seeding``): realization k is fully
determined by a child stream keyed to the GLOBAL index k, invariant to MPI layout / batch partition.
Use the SynHydro public API rather than reimplementing seeding:

    KirschGenerator.generate(..., realization_indices=[k0, k1, ...], seed=root_seed)  # keyed by global idx
    NowakDisaggregator.disaggregate(monthly_ensemble, seed=root_seed)                 # global idx = int key

The implementation is pywrdrb-coupled (Nowak downstream nodes, catchment-inflow recovery) and lives
in ``NYCOptimization/src/ensemble_generation.py``; this module stays the optimization-independent
contract (config + manifest + seeding).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from .manifest import EnsembleManifest


@dataclass(frozen=True)
class ForcingEnsembleConfig:
    """Configuration for one forcing-ensemble (or candidate-pool) generation.

    Attributes:
        root_seed: Root seed; per-realization streams are keyed to the global realization index.
            Derive it with :func:`scengen.seeds.design_seed` so distinct designs/draws are disjoint.
        n_forcing_profiles: Number of theta profiles (N_Theta). For a ``stationary`` population there
            is no theta, so this is simply the ensemble cardinality N (and
            ``realizations_per_profile`` must be 1).
        realizations_per_profile: Realizations generated per forcing profile (n_per_theta).
        realization_years: Disjoint window length L (years); must exceed the longest within-window
            drought (check vs the 1960s DRB drought of record ~4-5 yr).
        population: ``"stationary"`` (Kirsch-Nowak on the historic record, no climate perturbation)
            or ``"du_forced"`` (theta drawn from the CMIP6 harmonic hypercube).
        generator: Flow-generator family — ``"kn"`` (Kirsch monthly + Nowak daily disaggregation;
            every search-side design) or ``"hmm"`` (multi-site Gaussian-mixture HMM on annual flows,
            Gold et al. 2024, then Nowak annual->monthly->daily). ``"hmm"`` exists so the held-out
            test ensemble can carry a STRUCTURALLY DIFFERENT generator: a measuring stick built with
            the same generator the search ensembles use cannot detect a generator-structure
            dependence in the result.
        theta_sampler: How the harmonic-parameter box is sampled — ``"iid"`` (independent uniform
            draws; REQUIRED for a candidate pool that is later subsampled, and for probabilistic
            designs whose reference measure is plain Monte Carlo) or ``"lhs"`` (space-filling; valid
            only where the drawn sample itself is the design and is never subsampled — the search-side
            ``input_stratified`` design, and the held-out test ensemble, which is never subsampled and
            is never a control). Ignored for a ``stationary`` population.
        seed_domain: Namespace of ``root_seed`` (:func:`scengen.seeds.design_seed`), recorded in the
            staged ``_meta.json``. Consumed by NYCOptimization's import-time guard that a search
            ensemble and the test ensemble never share a seed stream (selection bias, Bonham et al.
            2024) — the guard compares recorded domains, because two ensembles can share a generator
            stream under different slugs.
        compute_hazard_image: If True, stream the candidate hazard image H (SSI-6 fit + POT pass per
            realization) and run the redundancy screen. Only the hazard-filling designs subsample, so
            this is pure waste for every other design; default False.
        flowtype: Historical full-record flow dataset key (e.g. ``pub_nhmv10_BC_withObsScaled``).
        baseline_period: (start, end) for the baseline-fit Kirsch (eqs 10-11 reference); used only
            when ``population == "du_forced"``.
        full_period: (start, end) for the full-record-fit Kirsch (the generator).
        output_dir: Staging directory (under STAGED_ENSEMBLE_DIR); artifacts land here.
        hazard_axes: Screened hazard-metric axis names to compute for H (from :mod:`hazard_metrics`).
        mean_frac_csv: CMIP6 multiplicative mean change-factor (``_frac_``) CSV (required for
            ``du_forced``; see :mod:`forcing_space`).
        variance_axis: If True, add the independent CMIP6-derived CV-change axis ``v_j`` so
            ``c_j = a_j * v_j`` (requires ``mean_abs_csv`` / ``std_csv``).
        mean_abs_csv, std_csv: Absolute monthly mean/std tables (for the CV-change envelope).
        bound_pct: Percentiles defining each harmonic parameter's CMIP6 range (default (5, 95), the
            empirical 90% range).
        margin: Optional fractional widening of the harmonic-parameter hypercube.
        start_date: Date assigned to day 0 of each realization.
        store_daily: If True (default), write the daily gage/inflow HDF5s. If False, discard daily
            traces after hazard computation and persist only H + params (the streaming mode).
        hazard_block_size: Forcing profiles processed per streaming block (bounds peak memory).
        chunk_size: When > 0 and < N, split the stored daily ensemble into contiguous chunks of
            ``chunk_size`` realizations, each written to a sibling staged dir ``{slug}__chunk{JJJ}``
            (must be a multiple of ``realizations_per_profile``). Only one chunk is resident at a
            time, so peak memory is bounded by the chunk. 0 (default) keeps a single directory.
    """

    root_seed: int
    n_forcing_profiles: int
    realizations_per_profile: int
    realization_years: int
    population: str = "du_forced"
    generator: str = "kn"
    theta_sampler: str = "iid"
    seed_domain: str | None = None
    compute_hazard_image: bool = False
    flowtype: str = "pub_nhmv10_BC_withObsScaled"
    baseline_period: tuple[str, str] = ("1980-10-01", "2019-09-30")
    full_period: tuple[str, str] = ("1945-10-01", "2022-09-30")
    output_dir: Path | None = None
    hazard_axes: tuple[str, ...] = ()
    mean_frac_csv: str | Path | None = None
    variance_axis: bool = False
    mean_abs_csv: str | Path | None = None
    std_csv: str | Path | None = None
    bound_pct: tuple[float, float] = (5.0, 95.0)
    margin: float = 0.0
    start_date: str = "1945-10-01"
    store_daily: bool = True
    hazard_block_size: int = 256
    chunk_size: int = 0
    extra: Mapping[str, Any] = field(default_factory=dict)

    @property
    def n_realizations(self) -> int:
        """Total ensemble cardinality N."""
        return self.n_forcing_profiles * self.realizations_per_profile


#: Valid ``ForcingEnsembleConfig.population`` values.
POPULATIONS: tuple[str, ...] = ("stationary", "du_forced")

#: Valid ``ForcingEnsembleConfig.generator`` values.
GENERATORS: tuple[str, ...] = ("kn", "hmm")


def _nycopt_impl(name: str):
    """Return the pywrdrb-coupled generation callable ``name`` from NYCOptimization.

    Kept as a lazy import so this module (the optimization-independent contract) still imports
    standalone without pywrdrb / the NYCOptimization package on the path.
    """
    try:
        from src import ensemble_generation
    except ImportError as exc:  # pragma: no cover - environment guard
        raise RuntimeError(
            "forcing-ensemble generation is implemented in NYCOptimization "
            "(src/ensemble_generation.py) and needs pywrdrb + that package importable "
            "(run from the NYCOptimization repo root with PYTHONPATH set)."
        ) from exc
    return getattr(ensemble_generation, name)


def child_seed_sequence(root_seed: int, n: int) -> list[np.random.SeedSequence]:
    """Return the ``n`` per-realization child seed sequences keyed by GLOBAL index.

    ``children[k]`` is the stream for global realization ``k`` and is invariant to how realizations
    are partitioned across MPI ranks/batches. This mirrors the canonical SynHydro helpers in
    ``synhydro.core.seeding`` (``spawn_realization_seed`` / ``realization_rng``); prefer those when
    generating, and especially ``spawn_realization_seed(root, k)`` for isolated regeneration (it
    reconstructs ``spawn(N)[k]`` from the spawn key without materializing all ``n`` children).
    """
    return list(np.random.SeedSequence(root_seed).spawn(n))


def generate_forcing_ensemble(config: ForcingEnsembleConfig) -> EnsembleManifest:
    """Generate an ensemble (or candidate pool), optionally stream H, and persist the manifest.

    Steps (methods 3.2):
        1. Fit the full-record Kirsch generator + Nowak disaggregator (+ the baseline-period Kirsch
           reference when ``population == "du_forced"``).
        2. ``du_forced``: for each forcing profile theta_i, derive (mean_period_new, std_period_new)
           via ``forcing_space.apply_climate_adjustment`` and generate
           ``realizations_per_profile`` realizations, each driven by its global-index child RNG.
           ``stationary``: skip the adjustment entirely and generate one realization per index from
           the full-record moments.
        3. For each realization: Nowak daily disaggregation + per-realization KDE downstream fill +
           ``_subtract_upstream_catchment_inflows``; optionally accumulate h(.) (methods 3.3).
        4. Persist generator params, the manifest, and (when ``store_daily``) the daily HDF5s.

    Returns:
        The :class:`EnsembleManifest` describing the generated ensemble.

    Note:
        Thin delegator to the pywrdrb-coupled NYCOptimization implementation.
    """
    return _nycopt_impl("generate_forcing_ensemble")(config)


def regenerate_realization(
    root_seed: int,
    global_index: int,
    *,
    config: ForcingEnsembleConfig,
) -> pd.DataFrame:
    """Regenerate a single realization bit-for-bit from its global index.

    Uses the deterministic SynHydro per-realization-RNG API:
    ``KirschGenerator.generate(realization_indices=[global_index], seed=root_seed)`` then
    ``NowakDisaggregator.disaggregate(monthly, seed=root_seed)`` on the key-``global_index`` ensemble,
    then the per-realization KDE downstream fill -- identical regardless of N or MPI layout. Validated
    by an end-to-end determinism regression test (generate the realization under two partitions; assert
    array-equal through the KDE fill, post float32/HDF5).

    Returns:
        Daily multi-node flows for the requested realization.

    Note:
        Thin delegator to the pywrdrb-coupled NYCOptimization implementation (see
        :func:`generate_forcing_ensemble`).
    """
    return _nycopt_impl("regenerate_realization")(root_seed, global_index, config=config)
