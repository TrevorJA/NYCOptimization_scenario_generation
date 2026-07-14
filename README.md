# NYCOptimization_scenario_generation

Optimization-independent **streamflow scenario-ensemble generation** and **ensemble-quality
diagnostics** for the NYC reservoir-operation scenario-design study. Companion to
`../NYCOptimization`; methodology in `../NYCOptimization/docs/notes/methods/scenario_design_methods.md`.

This repo produces the **forcing ensembles** (including the **candidate pool** a hazard-filling
design selects from) and their **hazard image**, runs the per-design **selectors**, and computes
**ensemble-quality (build-QC) diagnostics**. It is deliberately separate from the optimization repo
so generation can be developed and *vetted* in isolation; vetted ensembles are handed off, not
copied ad hoc.

## Two design families: generate-to vs select-from

The distinction is conceptual, and it decides which code path a design uses:

- **Input-space stratification GENERATES.** The forcing parameters `theta` are a knob on the
  generator, so the design draws an LHS over `theta` and generates one realization per design point.
  It never subsamples — there is nothing to snap to. Lives in `forcing_space` /
  `forcing_ensemble`.
- **Hazard-filling SELECTS.** Hazard coordinates (drought deficit volume, flood peak magnitude, …)
  are *emergent* properties of a realized flow sequence — no generator can be asked to produce a
  realization at a prescribed drought severity. So a hazard-space design must select from a finite
  **candidate pool**, and its LHS anchors must snap to the nearest real pool member. That
  nearest-neighbor snap is the whole reason `subsample` exists.

The selector is deterministic given its seed (LHS + NN-snap). There is no simulated annealing and no
discrepancy objective, which is what keeps L2-star discrepancy an *independent* build-QC gate.

## What lives here vs in NYCOptimization

| Here (`scengen`) | In `../NYCOptimization` |
|---|---|
| `forcing_space` — CMIP6 envelope + LHS/i.i.d. fill -> `theta` (the generate-to design) | `EnsembleSpec` / `ScenarioDesign` registry + `resolve_search_spec` dispatch |
| `forcing_ensemble` — Kirsch–Nowak generation, streaming hazard image | slug grammar + `register_ensemble_path` (the contract) |
| `hazard_metrics` — reuse + re-screen MOEA-FIND metrics | `ensemble_prep` staging into pywrdrb preprocessors |
| `subsample` / `hazard_filling` — hazard-space LHS + nearest-neighbor selector (the select-from design) | staging of the selected realizations + in-loop re-index hook |
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

**Hazard-filling is implemented and tested:** `subsample.py` (LHS + nearest-neighbor selector, in
rank space and — as a retained non-campaign sensitivity — absolute magnitude space, plus the copied
coverage/LHS primitives and the random baseline), `hazard_metrics.py` (copied MOEA-FIND event
descriptors; SynHydro provides SSI), and `hazard_filling.py` (driver:
`select_from_candidate_image` — Olden & Poff redundancy screen → tail-balanced axis set → selection,
operating on the streamed candidate hazard image, never on the pool timeseries). Run `pytest` for
the selector tests (pure numpy/scipy); the SSI-dependent tests require SynHydro and are skipped
otherwise.

No code is imported from MOEA-FIND; the useful pieces are **copied** here (MOEA-FIND may be deleted).
See `../NYCOptimization/docs/notes/methods/scenario_design_methods.md` for the specification.
Outputs are gitignored and regenerable; never commit HDF5/CSV.
