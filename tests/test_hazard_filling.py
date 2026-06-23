"""Tests for the hazard-filling driver. Skipped when SynHydro is unavailable."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("synhydro", reason="hazard_filling driver needs SynHydro's SSI")

from scengen import hazard_filling as hf  # noqa: E402
from scengen import hazard_metrics as hm  # noqa: E402
from synhydro.core.ensemble import Ensemble  # noqa: E402


def _monthly(n_years, *, seed, scale=100.0):
    rng = np.random.default_rng(seed)
    return rng.gamma(shape=2.0, scale=scale, size=n_years * 12)


def _daily(n_years, *, seed, scale=100.0):
    rng = np.random.default_rng(seed + 1000)
    return rng.gamma(shape=2.0, scale=scale, size=n_years * 365)


def test_build_subset_basic():
    ref_m = _monthly(78, seed=0)
    ref_d = _daily(78, seed=0)
    scen_m = np.vstack([_monthly(5, seed=s) for s in range(60)])
    scen_d = np.vstack([_daily(5, seed=s) for s in range(60)])
    out = hf.build_hazard_filling_subset(scen_m, scen_d, ref_m, ref_d, n=10, seed=0)
    assert out["selected_rows"].shape == (10,)
    assert set(out["selected_rows"].tolist()) <= set(range(60))
    assert out["candidate_axes"] == list(hm.CANDIDATE_EVENT_METRICS)
    assert out["H_candidates"].shape == (60, 8)
    # tail-balanced chosen set: <=2 dry + <=2 wet (<=4 total), subset of candidates.
    assert 1 <= len(out["chosen_axes"]) <= 4
    assert set(out["chosen_axes"]) <= set(hm.CANDIDATE_EVENT_METRICS)
    assert sum(a.startswith("drought") for a in out["chosen_axes"]) <= 2
    assert sum(a.startswith("flood") for a in out["chosen_axes"]) <= 2
    assert {"selected_L2_star", "random_L2_star"} <= set(out["coverage"])


def _toy_pool(tmp_path, n_real=6, n_days=60):
    """Write a tiny pywrdrb-style pool (gage + catchment HDF5s) and return its dir."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    nodes = list(hm.DEFAULT_NYC_INFLOW_NODES) + ["delMontague"]
    dates = pd.date_range("1945-10-01", periods=n_days, freq="D")
    rng = np.random.default_rng(0)
    for fname in ("gage_flow_mgd.hdf5", "catchment_inflow_mgd.hdf5"):
        by_real = {
            r: pd.DataFrame(rng.gamma(2.0, 50.0, size=(n_days, len(nodes))),
                            index=dates, columns=nodes)
            for r in range(n_real)
        }
        Ensemble(by_real).to_hdf5(str(tmp_path / fname))
    return tmp_path


def test_stage_subset_ensemble_roundtrip(tmp_path):
    pool_dir = _toy_pool(tmp_path / "pool")
    out_dir = tmp_path / "hazfill_5yr_n3_s0"
    selected = [4, 1, 5]
    meta = {"slug": "hazfill_5yr_n3_s0", "n_realizations": 3, "realization_years": 5}

    hf.stage_subset_ensemble(pool_dir, out_dir, selected, meta=meta)

    # _meta.json round-trips.
    back = json.loads((out_dir / "_meta.json").read_text())
    assert back["n_realizations"] == 3
    assert back["realization_years"] == 5

    # Reduced HDF5s exist and hold exactly the selected realizations, renumbered.
    for fname in ("gage_flow_mgd.hdf5", "catchment_inflow_mgd.hdf5"):
        red = Ensemble.from_hdf5(str(out_dir / fname), stored_by_node=True)
        assert sorted(red.data_by_realization) == [0, 1, 2]
        pool = Ensemble.from_hdf5(str(pool_dir / fname), stored_by_node=True)
        # new realization 0 == pool realization 4 (first selected id).
        np.testing.assert_allclose(
            red.data_by_realization[0].to_numpy(),
            pool.data_by_realization[4].to_numpy(),
        )
