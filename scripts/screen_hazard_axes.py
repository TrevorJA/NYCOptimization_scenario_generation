"""screen_hazard_axes.py - screen the candidate hazard-axis set for a staged pool.

Reads the candidate hazard image streamed at candidate-pool generation
(``hazard_image.npz``; the 8-axis candidate set — 5 dry SSI-6 controlling-event
run-theory axes + 3 wet POT axes, see :mod:`scengen.hazard_metrics`) and runs
the axis screen used by the live selection driver:

    1. per-metric spread      - drop degenerate axes (near-zero spread)
    2. near-duplicate dedupe  - prune groups connected at |rho_S| >= threshold
                                (default 0.95) to one canonical member; retain
                                every other non-degenerate axis

The Spearman correlation structure (matrix heatmap) and the PCA effective
dimension of the retained set are reported as diagnostics alongside the screen
— they never reduce the axis set further.

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
from scengen.hazard_filling import DEDUPE_RHO_THRESHOLD, screen_hazard_axes  # noqa: E402

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
    """Broken-stick scree plot of the retained axis set's effective dimensionality."""
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
    p.add_argument("--dedupe-threshold", type=float, default=DEDUPE_RHO_THRESHOLD)
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

    screen = screen_hazard_axes(H, axes, dedupe_threshold=args.dedupe_threshold)
    retained = screen["retained"]
    # Effective dimension of the RETAINED axes (what actually gets filled).
    ret_idx = [axes.index(a) for a in retained]
    pca = dg.pca_effective_dimension(H[:, ret_idx]) if len(retained) > 1 else {
        "explained_variance_ratio": np.array([1.0]), "broken_stick": np.array([1.0]),
        "effective_dimension": 1,
    }

    out_dir = args.out_dir or (args.ensemble_dir / "axis_screen")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n[screen] per-metric spread:")
    for a in axes:
        s = screen["spread"][a]
        flag = "  DROP (degenerate)" if s["degenerate"] else "  keep"
        print(f"  {a:24s} std={s['std']:.3g} iqr={s['iqr']:.3g} "
              f"skew={s['skew']:.2f} zero_frac={s['zero_frac']:.2f}{flag}")
    print(f"\n[screen] near-duplicate groups (|rho_S|>={args.dedupe_threshold:g}):")
    if screen["near_duplicate_groups"]:
        for grp in screen["near_duplicate_groups"]:
            kept = next(a for a in grp if a in retained)
            print(f"  {grp}  -> keep '{kept}'")
    else:
        print("  (none)")
    print(f"\n[screen] RETAINED axis set (m={len(retained)}): {retained}")
    pr = pca.get("participation_ratio", float(pca["effective_dimension"]))
    print(f"[screen] effective dimension of retained set: participation-ratio = {pr:.2f} "
          f"(EVR={np.round(pca['explained_variance_ratio'], 3).tolist()})")
    if pr > 0:
        for n in (100, 200):
            print(f"[screen]   N={n} gives ~{n ** (1.0 / pr):.1f} levels/axis at PR={pr:.2f}")

    summary = {
        "ensemble_dir": str(args.ensemble_dir),
        "candidate_axes": axes,
        "screen": {k: v for k, v in screen.items() if k != "spread"},
        "spread": screen["spread"],
        "effective_dimension": pca["effective_dimension"],
        "participation_ratio": pr,
        "explained_variance_ratio": np.asarray(pca["explained_variance_ratio"]).tolist(),
    }
    (out_dir / "axis_screen_summary.json").write_text(json.dumps(summary, indent=2))

    plot_correlation(np.asarray(screen["spearman_rho"]), screen["spearman_axes"],
                     out_dir / "axis_correlation.png")
    plot_scree(pca, out_dir / "effective_dimension.png")
    print(f"\n[screen] wrote summary + 2 figures -> {out_dir}")


if __name__ == "__main__":
    main()
