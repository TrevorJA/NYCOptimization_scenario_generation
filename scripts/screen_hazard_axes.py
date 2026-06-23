"""screen_hazard_axes.py - choose a low-redundancy wet+dry hazard-axis set.

Computes the 6-candidate run-theory event-descriptor hazard image (3 drought +
3 flood axes; :mod:`scengen.hazard_metrics`) on a staged master pool, then runs
the Olden & Poff (2003) screening pipeline to recommend a parsimonious,
low-redundancy axis set spanning both tails:

    1. per-metric spread     - drop degenerate axes (near-constant, zero-IQR,
                               heavy skew, or a >50% point mass at one value,
                               e.g. short-window event metrics that are 0 when a
                               scenario has no critical event)
    2. Spearman clustering   - cluster axes at |rho_S| >= 0.7; keep one (widest-
                               IQR) representative per cluster
    3. PCA broken-stick      - effective dimensionality m_eff = the number of
                               axes the space-filling subsample must fill, which
                               sets the N >= q^m sampling budget

Reads the staged pool's HDF5 (via SynHydro) and the historical reference inflow
CSV directly (no pywrdrb dependency). Exploratory: rerun while iterating the
candidate definitions / timescales.

Usage (from the scenario-generation repo root)::

    python scripts/screen_hazard_axes.py \
        --pool-dir ../NYCOptimization/outputs/synthetic_ensembles/kn_5yr_n200
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from scengen import diagnostics as dg  # noqa: E402
from scengen import hazard_metrics as hm  # noqa: E402
from scengen.hazard_filling import (  # noqa: E402
    daily_to_monthly,
    load_ensemble_daily_aggregate,
    load_ensemble_monthly_aggregate,
)

_DEFAULT_POOL = "../NYCOptimization/outputs/synthetic_ensembles/kn_5yr_n200"
_DEFAULT_REFERENCE_CSV = (
    "../Pywr-DRB/src/pywrdrb/data/flows/pub_nhmv10_BC_withObsScaled/"
    "catchment_inflow_mgd.csv"
)


def _load_reference(csv_path: Path, nodes) -> tuple[np.ndarray, np.ndarray]:
    """Historical (monthly, daily) aggregate NYC inflow from the reference CSV."""
    Q = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    daily = Q.loc[:, list(nodes)].sum(axis=1)
    monthly = daily_to_monthly(daily, agg="mean")
    return monthly, daily.to_numpy(dtype=float)


def plot_correlation(rho, axes, out_path: Path) -> None:
    fig, a = plt.subplots(figsize=(5.6, 4.8))
    im = a.imshow(np.abs(rho), vmin=0, vmax=1, cmap="magma_r")
    a.set_xticks(range(len(axes)))
    a.set_yticks(range(len(axes)))
    a.set_xticklabels(axes, rotation=45, ha="right", fontsize=7)
    a.set_yticklabels(axes, fontsize=7)
    for i in range(len(axes)):
        for j in range(len(axes)):
            a.text(j, i, f"{rho[i, j]:.2f}", ha="center", va="center",
                   fontsize=6, color="white" if abs(rho[i, j]) > 0.5 else "black")
    a.set_title("Spearman |rho| between candidate hazard axes", fontsize=10)
    fig.colorbar(im, ax=a, fraction=0.046)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_scree(pca, out_path: Path) -> None:
    evr, bs = pca["explained_variance_ratio"], pca["broken_stick"]
    x = np.arange(1, len(evr) + 1)
    fig, a = plt.subplots(figsize=(5.2, 3.6))
    a.bar(x, evr, color="0.7", label="explained variance")
    a.plot(x, bs, "o-", color="#c1272d", label="broken-stick")
    a.axvline(pca["effective_dimension"] + 0.5, ls="--", color="k",
              label=f"m_eff = {pca['effective_dimension']}")
    a.set_xlabel("principal component", fontsize=9)
    a.set_ylabel("variance fraction", fontsize=9)
    a.set_title("Effective hazard dimensionality (PCA broken-stick)", fontsize=10)
    a.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pool-dir", type=Path, default=Path(_DEFAULT_POOL))
    p.add_argument("--reference-csv", type=Path, default=Path(_DEFAULT_REFERENCE_CSV))
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--dry-timescale", type=int, default=6,
                   help="Drought SSI accumulation in months (SSI-6 reservoir default).")
    p.add_argument("--flood-threshold-pct", type=float, default=95.0,
                   help="POT high-flow threshold percentile of reference daily flow (95 = Q5).")
    p.add_argument("--redundancy-threshold", type=float, default=0.7)
    args = p.parse_args()

    pool_hdf5 = args.pool_dir / "catchment_inflow_mgd.hdf5"
    if not pool_hdf5.exists():
        raise SystemExit(f"pool HDF5 not found: {pool_hdf5}")
    nodes = hm.DEFAULT_NYC_INFLOW_NODES

    ref_monthly, ref_daily = _load_reference(args.reference_csv, nodes)
    pool_monthly, ids = load_ensemble_monthly_aggregate(pool_hdf5, nodes, agg="mean")
    pool_daily, _ = load_ensemble_daily_aggregate(pool_hdf5, nodes)
    print(f"[screen] pool={pool_monthly.shape[0]} scenarios "
          f"(months={pool_monthly.shape[1]}, days={pool_daily.shape[1]}); "
          f"reference months={len(ref_monthly)} days={len(ref_daily)}")

    H, axes = hm.compute_candidate_hazard_image(
        pool_monthly, pool_daily, ref_monthly, ref_daily,
        dry_timescale=args.dry_timescale, flood_threshold_pct=args.flood_threshold_pct,
    )

    # --- Olden & Poff screen ------------------------------------------------
    from scengen.hazard_filling import DEFAULT_AXIS_PRIORITY, select_balanced_axes

    spread = dg.per_metric_spread(H, axes)
    kept = [a for a in axes if not spread[a]["degenerate"]]
    keep_idx = [axes.index(a) for a in kept]
    clusters = dg.spearman_clusters(
        H[:, keep_idx], kept, threshold=args.redundancy_threshold,
        priority=DEFAULT_AXIS_PRIORITY,
    )
    recommended = select_balanced_axes(
        clusters["representatives"], DEFAULT_AXIS_PRIORITY, max_per_tail=2
    )
    # Effective dimension of the CHOSEN axes (what actually gets filled), not of
    # the full redundant candidate set (whose spectrum is flattened by redundancy).
    rec_idx = [axes.index(a) for a in recommended]
    pca = dg.pca_effective_dimension(H[:, rec_idx]) if len(recommended) > 1 else {
        "explained_variance_ratio": np.array([1.0]), "broken_stick": np.array([1.0]),
        "effective_dimension": 1,
    }

    out_dir = args.out_dir or (args.pool_dir / "axis_screen")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n[screen] per-metric spread (degenerate axes dropped):")
    for a in axes:
        s = spread[a]
        flag = "  DROP" if s["degenerate"] else "  keep"
        print(f"  {a:24s} std={s['std']:.3g} iqr={s['iqr']:.3g} "
              f"skew={s['skew']:.2f} zero_frac={s['zero_frac']:.2f}{flag}")
    print(f"\n[screen] surviving axes: {kept}")
    print(f"[screen] redundancy clusters (|rho_S|>={args.redundancy_threshold}):")
    for c, rep in zip(clusters["clusters"], clusters["representatives"]):
        print(f"  {c}  -> keep '{rep}'")
    print(f"\n[screen] cluster representatives: {clusters['representatives']}")
    print(f"[screen] RECOMMENDED tail-balanced axis set (<=2 dry + <=2 wet): {recommended}")
    pr = pca.get("participation_ratio", float(pca["effective_dimension"]))
    print(f"[screen] effective dimension of chosen set: participation-ratio = {pr:.2f} "
          f"(EVR={np.round(pca['explained_variance_ratio'], 3).tolist()})")
    # N >= q^m budget note for the CHOSEN axis set (what actually gets filled).
    m_fill = len(recommended)
    for q in (3, 4):
        print(f"[screen]   to fill m={m_fill} (chosen) at q={q} levels/axis: "
              f"N >= {q ** m_fill}")
    if pr > 0:
        n64_q = 64 ** (1.0 / pr)
        print(f"[screen]   N=64 gives ~{n64_q:.1f} levels/axis at PR={pr:.2f}")

    summary = {
        "pool": str(args.pool_dir),
        "candidate_axes": axes,
        "spread": spread,
        "surviving_axes": kept,
        "clusters": clusters["clusters"],
        "recommended_axes": recommended,
        "effective_dimension": pca["effective_dimension"],
        "explained_variance_ratio": pca["explained_variance_ratio"].tolist(),
    }
    (out_dir / "axis_screen_summary.json").write_text(json.dumps(summary, indent=2))

    plot_correlation(clusters["rho"], kept, out_dir / "axis_correlation.png")
    plot_scree(pca, out_dir / "effective_dimension.png")
    print(f"\n[screen] wrote summary + 2 figures -> {out_dir}")


if __name__ == "__main__":
    main()
