"""Tests for the hazard-filling driver. Skipped when SynHydro is unavailable."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("synhydro", reason="hazard_filling driver needs SynHydro's SSI")

from scengen import hazard_filling as hf  # noqa: E402
from scengen import hazard_metrics as hm  # noqa: E402
from scengen.manifest import EnsembleManifest  # noqa: E402


def _monthly(n_years, *, seed, scale=100.0):
    rng = np.random.default_rng(seed)
    return rng.gamma(shape=2.0, scale=scale, size=n_years * 12)


def test_build_subset_basic():
    ref = _monthly(78, seed=0)
    scenarios = np.vstack([_monthly(5, seed=s) for s in range(60)])
    out = hf.build_hazard_filling_subset(scenarios, ref, n=10, seed=0)
    assert out["selected_rows"].shape == (10,)
    assert set(out["selected_rows"].tolist()) <= set(range(60))
    assert out["hazard_axes"] == list(hm.PRIMARY_METRICS)
    assert out["H"].shape == (60, 3)
    assert {"selected_L2_star", "random_L2_star"} <= set(out["coverage"])


def test_write_subset_manifest_roundtrip(tmp_path):
    ref = _monthly(78, seed=0)
    scenarios = np.vstack([_monthly(5, seed=s) for s in range(40)])
    out = hf.build_hazard_filling_subset(scenarios, ref, n=8, seed=1)
    path = tmp_path / "hazard_filling_n8_seed1.json"
    hf.write_subset_manifest(
        path,
        master_slug="kn_5yr_n40",
        selected_global_indices=out["selected_rows"].tolist(),
        hazard_axes=out["hazard_axes"],
        realization_years=5,
        seed=1,
        coverage=out["coverage"],
    )
    back = EnsembleManifest.from_json(path)
    assert back.design == "hazard_filling"
    assert back.slug == "kn_5yr_n40"
    assert len(back.realization_global_indices) == 8
    assert back.hazard_axes == list(hm.PRIMARY_METRICS)
