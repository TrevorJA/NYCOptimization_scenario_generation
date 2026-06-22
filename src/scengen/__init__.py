"""scengen — streamflow scenario-ensemble generation and ensemble-quality diagnostics.

Optimization-independent generation for the NYC reservoir scenario-design study. See
``../NYCOptimization/docs/notes/methods/scenario_design_methods.md`` for the specification;
each module docstring cites the relevant section.

Public surface (stubs):
    forcing_space   - CMIP6 envelope + LHS-fill -> theta forcing profiles (methods 3.1)
    master_ensemble - Kirsch-Nowak master ensemble + streaming hazard image (methods 3.2)
    hazard_metrics  - reuse + re-screen MOEA-FIND hazard metrics (methods 3.3)
    subsample       - hazard-filling and support-point selectors (methods 4.6, 4.6.1)
    diagnostics     - ensemble-quality / build-QC diagnostics (methods 6a)
    manifest        - provenance manifest emitted with each staged ensemble (handoff contract)
"""

from __future__ import annotations

__all__ = [
    "forcing_space",
    "master_ensemble",
    "hazard_metrics",
    "subsample",
    "diagnostics",
    "manifest",
]

__version__ = "0.0.1"
