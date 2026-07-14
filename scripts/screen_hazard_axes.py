"""screen_hazard_axes.py - choose a low-redundancy wet+dry hazard-axis set.

Reads the candidate hazard image streamed at candidate-pool generation
(``hazard_image.npz``; 3+ drought and 3+ flood run-theory event descriptors, see
:mod:`scengen.hazard_metrics`) and runs the Olden & Poff (2003) screening
pipeline to recommend a parsimonious, low-redundancy axis set spanning both
tails:

    1. per-metric spread     - drop degenerate axes (near-constant, zero-IQR,
                               heavy skew, or a >50% point mass at one value,
                               e.g. short-window event metrics that are 0 when a
                               scenario has no critical event)
    2. Spearman clustering   - cluster axes at |rho_S| >= 0.7; keep one (widest-
                               IQR) representative per cluster
    3. PCA broken-stick      - effective dimensionality m_eff = the number of
                               axes the space-filling subsample must fill, which
                               sets the N >= q^m sampling budget

Operates on the hazard image alone -- no pool timeseries, no SynHydro, no
pywrdrb -- so it scales to a very large candidate pool. Exploratory: rerun while
iterating the candidate metric definitions / timescales.

Usage (from the scenario-generation repo root)::

    python scripts/screen_hazard_axes.py \
        --ensemble-dir ../NYCOptimization/outputs/synthetic_ensembles/hazfill_5yr_n64_s0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from scengen import diagnostics as dg  # noqa: E402
from scengen.hazard_filling import DEFAULT_AXIS_PRIORITY, select_balanced_axes  # noqa: E402

_DEFAULT_ENSEMBLE = "../NYCOptimization/outputs/synthetic_ensembles/hazfill_5yr_n64_s0"


def plot_correlation(rho: np.ndarray, axes: list[str], out_path: Path) -> None:
    """Heatmap of the Spearman |rho| matrix between the surviving candidate axes."""
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


def plot_scree(pca: dict, out_path: Path) -> None:
    """Broken-stick scree plot of the chosen axis set's effective dimensionality."""
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
    p.add_argument("--ensemble-dir", type=Path, default=Path(_DEFAULT_ENSEMBLE),
                   help="Directory holding the pool's hazard_image.npz.")
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--redundancy-threshold", type=float, default=0.7)
    args = p.parse_args()

    img_path = args.ensemble_dir / "hazard_image.npz"
    if not img_path.exists():
        raise SystemExit(
            f"hazard_image.npz not found in {args.ensemble_dir}. Generate the candidate "
            f"pool first (NYCOptimization step 02/03)."
        )
    img = dg.load_hazard_image(img_path)
    H, axes = img["H"], img["hazard_axes"]
    print(f"[screen] pool={H.shape[0]} scenarios, {H.shape[1]} candidate axes")

    # --- Olden & Poff screen ------------------------------------------------
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

    out_dir = args.out_dir or (args.ensemble_dir / "axis_screen")
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
        "ensemble_dir": str(args.ensemble_dir),
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
