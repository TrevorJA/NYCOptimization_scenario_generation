"""Provenance manifest emitted with each staged ensemble (handoff contract).

The generation repo emits, alongside the staged pywrdrb-format HDF5, an ``_meta.json`` manifest so
``../NYCOptimization`` can resolve and trust a staged ensemble by slug. The manifest is the
authoritative record of *how* an ensemble was constructed; it is what makes "copy once vetted" a
versioned, audit-able handoff rather than a manual file copy.

See ``scenario_design_methods.md`` §7 (handoff contract) and §4.7 (test-ensemble provenance).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class EnsembleManifest:
    """Immutable provenance record for one staged scenario ensemble.

    Attributes:
        design: Scenario-design name (e.g. ``"hazard_filling"``); matches ``ScenarioDesign.name``.
        draw: Independent construction draw index (selector/ensemble-draw replicate; 0-based).
        n_realizations: Number of scenarios in the staged ensemble.
        realization_years: Scenario length in years (window length L).
        master_seed: Master seed for the deterministic generator (global-index RNG streams).
        realization_global_indices: Global indices into the master ensemble for each staged
            realization, so any realization regenerates bit-for-bit from ``master_seed`` + index.
        forcing_hash: Stable hash of the forcing-space configuration (CMIP6 envelope + LHS settings).
        hazard_axes: Names of the screened hazard-metric axes used for selection.
        screen_result: Serialized redundancy-screen output (retained axes, clusters, VIF/PCA).
        slug: Output slug under which NYCOptimization resolves this ensemble.
        created: ISO timestamp (passed in by the caller; this module performs no clock reads).
        source_kind: Generator family tag (e.g. ``"synhydro_kn"``).
        start_date: Date of day 0 of every staged realization (a December 1;
            the generator synthesizes calendar-year sequences, so one extra
            year is generated and the frames trimmed to this epoch during
            generation — the index anchors here by construction).
        notes: Free-form provenance notes.
    """

    design: str
    draw: int
    n_realizations: int
    realization_years: int
    master_seed: int
    realization_global_indices: Sequence[int]
    forcing_hash: str
    hazard_axes: Sequence[str]
    screen_result: Mapping[str, Any] = field(default_factory=dict)
    slug: str = ""
    created: str = ""
    source_kind: str = "synhydro_kn"
    start_date: str = ""
    notes: str = ""

    def to_json(self) -> str:
        """Return the manifest as a pretty-printed JSON string."""
        return json.dumps(asdict(self), indent=2, sort_keys=True)

    def write(self, directory: str | Path, filename: str = "_meta.json") -> Path:
        """Write the manifest to ``directory/filename`` and return the path."""
        out = Path(directory) / filename
        out.write_text(self.to_json())
        return out

    @classmethod
    def from_json(cls, path: str | Path) -> "EnsembleManifest":
        """Load a manifest written by :meth:`write`."""
        data = json.loads(Path(path).read_text())
        return cls(**data)
