"""compare_sampling_spaces.py - characterize the absolute (distorted) vs CDF
(faithful) hazard-filling selection schemes, offline, on a staged hazard image.

Loads the candidate hazard image written beside a staged hazard-filling ensemble
(``hazard_image.npz``), subsets to the chosen axes, and runs three selectors at
the same N and seed:

    cdf            - LHS+NN in empirical-CDF (rank) space  [faithful / representative]
    absolute       - LHS+NN in absolute min-max space      [distorted / extreme-weighted]
    absolute_p1_99 - absolute with robust 1/99-pct bounds   [distorted, outlier-robust]

It reports, per scheme: marginal fidelity to the pool (KS distance, upper-tail
share, median), separation/degeneracy (nearest-neighbor stats, extreme-corner
concentration), and coverage cross-reported in BOTH spaces (each scheme wins in
its own space; the cross term shows the trade). Figures: per-axis marginal
density overlays and a 2-D magnitude-axis scatter.

This is a decision aid for whether to include the absolute (distorted) design as
a separate experimental arm. Pure numpy/scipy + the staged npz; no pywrdrb.

Usage::

    python scripts/compare_sampling_spaces.py \
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
from scipy.stats import ks_2samp, spearmanr  # noqa: E402

from scengen import diagnostics as dg  # noqa: E402
from scengen import subsample as ss  # noqa: E402

_DEFAULT = "../NYCOptimization/outputs/synthetic_ensembles/hazfill_5yr_n64_s0"
_SCHEME_COLORS = {"cdf": "#1f6fb4", "absolute": "#c1272d", "absolute_p1_99": "#e8920c"}


def _selectors(n, seed):
    return {
        "cdf": lambda H: ss.hazard_filling_subsample(H, n, seed=seed),
        "absolute": lambda H: ss.absolute_filling_subsample(H, n, seed=seed),
        "absolute_p1_99": lambda H: ss.absolute_filling_subsample(
            H, n, seed=seed, lo_pct=1.0, hi_pct=99.0
        ),
    }


def _mean_abs_spearman(A, m):
    r, _ = spearmanr(A)
    r = np.atleast_2d(r)
    iu = np.triu_indices(m, 1)
    return float(np.mean(np.abs(r[iu])))


def characterize(H, sel, axes):
    """Per-scheme diagnostics for a selection (absolute-space view + coverage)."""
    M, m = H.shape
    Xcdf = ss.empirical_cdf_normalize(H)
    Xabs = ss.minmax_normalize(H)
    lb, ub = np.zeros(m), np.ones(m)
    per_axis = {}
    for k, a in enumerate(axes):
        q90 = np.percentile(H[:, k], 90)
        per_axis[a] = {
            "ks_vs_pool": float(ks_2samp(H[sel, k], H[:, k]).statistic),
            "tail_share_p90": float(np.mean(H[sel, k] > q90)),
            "median_ratio": float(np.median(H[sel, k]) / (np.median(H[:, k]) + 1e-12)),
        }
    # extreme-corner concentration: fraction of selected above pool P90 on >=1 axis.
    p90 = np.percentile(H, 90, axis=0)
    frac_extreme = float(np.mean((H[sel] > p90).any(axis=1)))
    return {
        "per_axis": per_axis,
        "frac_in_any_p90_corner": frac_extreme,
        "mean_ks_vs_pool": float(np.mean([per_axis[a]["ks_vs_pool"] for a in axes])),
        "mean_tail_share_p90": float(np.mean([per_axis[a]["tail_share_p90"] for a in axes])),
        "nn_min_abs": ss.coverage_metrics(Xabs[sel], lb, ub).get("nn_min", 0.0),
        "nn_cv_abs": ss.coverage_metrics(Xabs[sel], lb, ub).get("nn_cv", 0.0),
        "L2star_cdf_space": ss.coverage_metrics(Xcdf[sel], lb, ub)["L2_star_discrepancy"],
        "L2star_abs_space": ss.coverage_metrics(Xabs[sel], lb, ub)["L2_star_discrepancy"],
        "mean_abs_spearman": _mean_abs_spearman(H[sel], m),
    }


def plot_marginals(H, sels, axes, out_path: Path) -> None:
    m = H.shape[1]
    ncol = 2
    nrow = int(np.ceil(m / ncol))
    fig, ax = plt.subplots(nrow, ncol, figsize=(5.4 * ncol, 3.2 * nrow), squeeze=False)
    for k, a in enumerate(axes):
        axx = ax[k // ncol][k % ncol]
        lo, hi = float(H[:, k].min()), float(H[:, k].max())
        axx.hist(H[:, k], bins=30, range=(lo, hi), density=True, color="0.8", label="pool")
        for name, sel in sels.items():
            axx.hist(H[sel, k], bins=30, range=(lo, hi), density=True, histtype="step",
                     lw=1.8, color=_SCHEME_COLORS[name], label=name)
        axx.set_title(a, fontsize=9)
        axx.set_yticks([])
        axx.tick_params(labelsize=7)
        if k == 0:
            axx.legend(fontsize=7)
    for k in range(m, nrow * ncol):
        ax[k // ncol][k % ncol].axis("off")
    fig.suptitle("Per-axis marginals: pool vs CDF (faithful) vs absolute (distorted)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_magnitude_scatter(H, sels, axes, out_path: Path) -> None:
    # First dry + first wet magnitude-style axes (fallback to first two).
    dry = next((i for i, a in enumerate(axes) if a.startswith("drought")), 0)
    wet = next((i for i, a in enumerate(axes) if a.startswith("flood")), 1)
    fig, ax = plt.subplots(1, len(sels), figsize=(4.0 * len(sels), 3.8), squeeze=False)
    for c, (name, sel) in enumerate(sels.items()):
        a = ax[0][c]
        a.scatter(H[:, dry], H[:, wet], s=8, c="0.8", edgecolors="none")
        a.scatter(H[sel, dry], H[sel, wet], s=30, c=_SCHEME_COLORS[name],
                  edgecolors="white", linewidths=0.4)
        a.set_title(name, fontsize=9)
        a.set_xlabel(axes[dry], fontsize=8)
        if c == 0:
            a.set_ylabel(axes[wet], fontsize=8)
        a.tick_params(labelsize=7)
    fig.suptitle("Where each scheme lands (dry vs wet magnitude, absolute units)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ensemble-dir", type=Path, default=Path(_DEFAULT))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", type=Path, default=None)
    args = p.parse_args()

    img = dg.load_hazard_image(args.ensemble_dir / "hazard_image.npz")
    axes = img["chosen_axes"]
    col = [img["hazard_axes"].index(a) for a in axes]
    H = img["H"][:, col]
    n = len(img["selected_rows"])
    M = H.shape[0]

    sels = {name: fn(H) for name, fn in _selectors(n, args.seed).items()}
    reports = {name: characterize(H, sel, axes) for name, sel in sels.items()}

    out_dir = args.out_dir or (args.ensemble_dir / "sampling_space_comparison")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[cmp] pool M={M}, n={n}, axes={axes}\n")
    hdr = f"{'scheme':16s} {'meanKS':>7s} {'meanTail>P90':>12s} {'%inP90corner':>13s} " \
          f"{'L2*cdf':>8s} {'L2*abs':>8s} {'nn_min_abs':>10s}"
    print(hdr)
    for name, r in reports.items():
        print(f"{name:16s} {r['mean_ks_vs_pool']:7.3f} {r['mean_tail_share_p90']:12.3f} "
              f"{r['frac_in_any_p90_corner']:13.3f} {r['L2star_cdf_space']:8.4f} "
              f"{r['L2star_abs_space']:8.4f} {r['nn_min_abs']:10.4f}")
    print("\n(meanTail>P90: unbiased ~ 0.10; higher = more extreme-weighted. "
          "%inP90corner: share of selected in >=1 axis's top decile.)")

    (out_dir / "sampling_space_comparison.json").write_text(json.dumps(
        {"axes": axes, "n": n, "M": M, "reports": reports}, indent=2))
    plot_marginals(H, sels, axes, out_dir / "marginals_by_scheme.png")
    plot_magnitude_scatter(H, sels, axes, out_dir / "magnitude_scatter_by_scheme.png")
    print(f"\n[cmp] wrote summary + 2 figures -> {out_dir}")


if __name__ == "__main__":
    main()
