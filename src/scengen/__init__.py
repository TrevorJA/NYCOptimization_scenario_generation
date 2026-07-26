"""scengen — streamflow scenario-ensemble generation and ensemble-quality diagnostics.

Optimization-independent generation for the NYC reservoir scenario-design study. See
``../NYCOptimization/docs/notes/methods/scenario_design_methods.md`` for the specification;
each module docstring cites the relevant section.

Public surface:
    forcing_space    - CMIP6 envelope + LHS/i.i.d. fill -> theta forcing profiles (methods 3.1).
                       Input-space stratification GENERATES one realization per LHS design point:
                       theta is a knob on the generator, so there is nothing to subsample.
    forcing_ensemble - Kirsch-Nowak ensemble / candidate pool + optional hazard image (methods 3.2)
    hazard_metrics   - 8-axis candidate event-descriptor hazard image (methods 3.3)
    subsample        - hazard-space LHS + nearest-neighbor selectors (methods 4.3). Hazard
                       coordinates are EMERGENT from a realized sequence, so a hazard design must
                       SELECT FROM a candidate pool and snap its LHS anchors to the nearest member.
                       Absolute (range-scaled) space is the campaign selector; rank space is the
                       retained non-campaign sensitivity.
    hazard_filling   - hazard-filling design driver: screen + select on a candidate hazard image
    seeds            - namespaced generator seeds (cross-design disjointness contract)
    diagnostics      - ensemble-quality / build-QC diagnostics (methods 6)
    manifest         - provenance manifest emitted with each staged ensemble (handoff contract)
"""

from __future__ import annotations

from .seeds import SEED_DOMAINS, design_seed

__all__ = [
    "forcing_space",
    "forcing_ensemble",
    "hazard_metrics",
    "subsample",
    "hazard_filling",
    "seeds",
    "diagnostics",
    "manifest",
    "SEED_DOMAINS",
    "design_seed",
]

__version__ = "0.0.1"
