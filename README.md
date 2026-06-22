# NYCOptimization_scenario_generation

Optimization-independent **streamflow scenario-ensemble generation** and **ensemble-quality
diagnostics** for the NYC reservoir-operation scenario-design study. Companion to
`../NYCOptimization`; methodology in `../NYCOptimization/docs/notes/methods/scenario_design_methods.md`.

This repo produces the **master ensemble** and its **hazard image**, runs the per-design
**subsampling** selectors, and computes **ensemble-quality (build-QC) diagnostics**. It is deliberately
separate from the optimization repo so generation can be developed and *vetted* in isolation; vetted
ensembles are handed off, not copied ad hoc.

## What lives here vs in NYCOptimization

| Here (`scengen`) | In `../NYCOptimization` |
|---|---|
| `forcing_space` — CMIP6 envelope + LHS-fill -> `theta` | `EnsembleSpec` / `ScenarioDesign` registry + `resolve_search_spec` dispatch |
| `master_ensemble` — Kirsch–Nowak generation, streaming hazard image | slug grammar + `register_ensemble_path` (the contract) |
| `hazard_metrics` — reuse + re-screen MOEA-FIND metrics | `ensemble_prep` staging into pywrdrb preprocessors |
| `subsample` — hazard-filling + support-point selectors | `resample_per_eval` in-loop re-index hook |
| `diagnostics` — coverage / redundancy / fidelity (§6a) | outcome diagnostics (§6b: overfitting gap, stability) |
| `manifest` — provenance manifest emitted with each staged ensemble | consumption of staged ensembles by slug |

## Handoff contract

The generator emits, under the shared staging directory:
- staged pywrdrb-format HDF5 ensembles (gage flows, catchment inflows), and
- a **provenance manifest** (`_meta.json`) carrying: scenario design, draw index, forcing-space hash,
  per-realization seeds, hazard-axis definitions + screen results, and the master-seed.

`../NYCOptimization` consumes these by slug via `EnsembleSpec` / `register_ensemble_path`. Prefer a
versioned/hashed stage over a manual copy (avoids provenance drift).

## Dependencies

- `../SynHydro` (Kirsch–Nowak; **requires** the per-realization-deterministic RNG upgrade — selected
  realizations must regenerate bit-for-bit from a stored seed).
- `../MOEA-FIND` (reused: `src/metrics/{drought_metrics,short_block,extended,screening}.py`,
  `src/discovery/analysis.py::{coverage_metrics, generate_lhs_samples}`).
- `../CMIP6_multimodel_streamflow` (the `_frac_` monthly change-factor CSV).

## Status

**Hazard-filling (initial draft) is implemented and tested:** `subsample.py` (stratified-maximin
selector + copied coverage/LHS primitives), `hazard_metrics.py` (copied MOEA-FIND "primary" SSI
metrics; SynHydro provides SSI), and `hazard_filling.py` (driver: staged pool → hazard image →
subset → manifest). Run `pytest` for the selector tests (pure numpy/scipy); the SSI-dependent tests
require SynHydro and are skipped otherwise. `forcing_space.py`, `master_ensemble.py`, and
`diagnostics.py` remain typed stubs.

No code is imported from MOEA-FIND; the useful pieces are **copied** here (MOEA-FIND may be deleted).
See `../NYCOptimization/docs/notes/methods/scenario_design_methods.md` for the specification.
Outputs are gitignored and regenerable; never commit HDF5/CSV.
