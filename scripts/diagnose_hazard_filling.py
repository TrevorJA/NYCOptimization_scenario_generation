"""diagnose_hazard_filling.py - exploratory build-QC diagnostics for a staged
hazard-filling ensemble.

Reads the ``hazard_image.npz`` that the NYCOptimization subsample step writes
beside a staged hazard-filling ensemble (the pool hazard image + the selected
rows), computes the coverage / build-QC diagnostics (methods 6), and renders a
small set of exploratory figures:

    1. hazard-space pair grid   - pool vs selected, normalized hazard space
    2. marginal coverage        - per-axis pool density + selected positions
    3. discrepancy-in-context   - selected L2-star vs the random-subset baseline

These are exploratory diagnostics meant to be iterated while the hazard metrics
and selector are still being settled; manuscript-grade figures will later live
in NYCOptimization. Numeric diagnostics come from :mod:`scengen.diagnostics`;
this script owns only the orchestration and plotting.

Usage (from the scenario-generation repo root)::

    python scripts/diagnose_hazard_filling.py \
        --ensemble-dir ../NYCOptimization/outputs/synthetic_ensembles/hazfill_5yr_n10_s0
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
from scengen.subsample import empirical_cdf_normalize  # noqa: E402

_DEFAULT_ENSEMBLE = (
    "../NYCOptimization/outputs/synthetic_ensembles/hazfill_5yr_n10_s0"
)
_POOL_COLOR = "0.75"
_SEL_COLOR = "#c1272d"

#: Physical units per candidate axis, for absolute-space axis labels.
_AXIS_UNITS = {
    "drought_magnitude": "|SSI|·mo",
    "drought_duration": "mo",
    "drought_severity": "|SSI|",
    "drought_onset_rate": "|SSI|/mo",
    "drought_recovery_rate": "|SSI|/mo",
    "flood_peak_discharge": "x mean daily",
    "flood_pulse_duration": "days",
    "flood_rise_rate": "x mean daily",
}


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def _diag_density(ax, pool_vals, sel_vals, value_range, *, bins: int = 22) -> None:
    """Overlay the POOL and SELECTED marginal densities on a diagonal panel.

    Both are drawn as density-normalized (area=1) distributions on the same
    axis so they are directly comparable: pool as a filled grey histogram,
    selected as a red step outline. No per-point rug lines. Coinciding curves
    mean the selected marginal reproduces the pool marginal (what rank-space
    filling gives); divergence into the upper tail is the deliberate
    over-representation the campaign selector administers.
    """
    ax.hist(pool_vals, bins=bins, range=value_range, density=True,
            color=_POOL_COLOR, label="pool")
    ax.hist(sel_vals, bins=bins, range=value_range, density=True,
            histtype="step", color=_SEL_COLOR, lw=1.6, label="selected")
    ax.set_yticks([])


def plot_hazard_pair_grid(X, sel, axes, out_path: Path) -> None:
    """Pair grid of the normalized hazard space: pool (grey) vs selected (red)."""
    m = X.shape[1]
    not_sel = np.setdiff1d(np.arange(len(X)), sel)
    fig, ax = plt.subplots(m, m, figsize=(2.6 * m, 2.6 * m), squeeze=False)
    for i in range(m):
        for j in range(m):
            a = ax[i][j]
            if i == j:
                _diag_density(a, X[:, i], X[sel, i], (0.0, 1.0))
            else:
                a.scatter(X[not_sel, j], X[not_sel, i], s=10, c=_POOL_COLOR,
                          label="pool", edgecolors="none")
                a.scatter(X[sel, j], X[sel, i], s=36, c=_SEL_COLOR,
                          label="selected", edgecolors="white", linewidths=0.5)
                a.set_xlim(0, 1)
                a.set_ylim(0, 1)
            if i == m - 1:
                a.set_xlabel(axes[j], fontsize=8)
            if j == 0:
                a.set_ylabel(axes[i], fontsize=8)
            a.tick_params(labelsize=7)
    fig.suptitle("Hazard space (empirical-CDF normalized): pool vs selected", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_hazard_pair_grid_raw(H, sel, axes, out_path: Path) -> None:
    """Pair grid in ABSOLUTE metric units: pool (grey) vs selected (red).

    Unlike :func:`plot_hazard_pair_grid` (which works in empirical-CDF rank
    space), this shows the raw, skewed metric distributions and where the
    selected subset lands in real units -- the view in which the campaign
    (absolute-space) selector's over-representation of the rare severe corners
    is visible.
    """
    H = np.asarray(H, dtype=float)
    m = H.shape[1]
    not_sel = np.setdiff1d(np.arange(len(H)), sel)
    labels = [f"{a}\n({_AXIS_UNITS.get(a, '')})" for a in axes]
    fig, ax = plt.subplots(m, m, figsize=(2.7 * m, 2.7 * m), squeeze=False)
    for i in range(m):
        for j in range(m):
            a = ax[i][j]
            if i == j:
                lo, hi = float(H[:, i].min()), float(H[:, i].max())
                _diag_density(a, H[:, i], H[sel, i], (lo, hi))
            else:
                a.scatter(H[not_sel, j], H[not_sel, i], s=10, c=_POOL_COLOR,
                          edgecolors="none")
                a.scatter(H[sel, j], H[sel, i], s=34, c=_SEL_COLOR,
                          edgecolors="white", linewidths=0.5)
            if i == m - 1:
                a.set_xlabel(labels[j], fontsize=8)
            if j == 0:
                a.set_ylabel(labels[i], fontsize=8)
            a.tick_params(labelsize=7)
    fig.suptitle("Hazard space (ABSOLUTE metric units): pool vs selected", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_marginal_coverage(X, sel, axes, gaps, out_path: Path) -> None:
    """Per-axis pool density with selected positions and the largest gap."""
    m = X.shape[1]
    fig, ax = plt.subplots(1, m, figsize=(3.4 * m, 3.0), squeeze=False)
    for a_idx in range(m):
        a = ax[0][a_idx]
        a.hist(X[:, a_idx], bins=20, range=(0, 1), color=_POOL_COLOR, density=True)
        pts = np.sort(X[sel, a_idx])
        a.plot(pts, np.full_like(pts, -0.15), "|", color=_SEL_COLOR, ms=14, mew=1.5)
        a.set_title(f"{axes[a_idx]}\nmax gap = {gaps[a_idx]:.3f}", fontsize=9)
        a.set_xlim(0, 1)
        a.set_xlabel("normalized hazard", fontsize=8)
        a.tick_params(labelsize=7)
    fig.suptitle("Marginal coverage per hazard axis (selected = red ticks)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_discrepancy_in_context(report, out_path: Path, *, seed: int = 0) -> None:
    """Selected L2-star against the bootstrap random-subset distribution."""
    rnd = dg.expected_random_discrepancy(
        report.n, report.m, n_boot=2000, seed=seed,
    )  # geometry reference + we overlay the pool-based mean from the report
    fig, a = plt.subplots(figsize=(5.5, 3.6))
    a.hist(rnd["samples"], bins=40, color=_POOL_COLOR, label="random subsets")
    a.axvline(report.l2_star, color=_SEL_COLOR, lw=2,
              label=f"selected = {report.l2_star:.4f}")
    a.axvline(report.random_l2_mean, color="k", ls="--", lw=1,
              label=f"random mean = {report.random_l2_mean:.4f}")
    a.set_xlabel("L2-star discrepancy (lower = more uniform)", fontsize=9)
    a.set_ylabel("count", fontsize=9)
    a.set_title(
        f"Coverage in context: selected beats "
        f"{(1 - report.discrepancy_percentile) * 100:.0f}% of random draws",
        fontsize=10,
    )
    a.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--ensemble-dir", type=Path, default=Path(_DEFAULT_ENSEMBLE),
        help="Staged hazard-filling ensemble dir (holds hazard_image.npz).",
    )
    p.add_argument(
        "--out-dir", type=Path, default=None,
        help="Figure output dir (default: <ensemble-dir>/diagnostics).",
    )
    p.add_argument("--seed", type=int, default=0, help="Bootstrap RNG seed.")
    p.add_argument("--n-boot", type=int, default=2000, help="Bootstrap replicates.")
    args = p.parse_args()

    img_path = args.ensemble_dir / "hazard_image.npz"
    if not img_path.exists():
        raise SystemExit(
            f"hazard_image.npz not found in {args.ensemble_dir}. Run the "
            f"NYCOptimization selection step (scripts/main/select_hazard_filling.py)."
        )
    img = dg.load_hazard_image(img_path)
    sel = img["selected_rows"]
    # Diagnose coverage on the screened (chosen) axes actually used for selection.
    axes = img["chosen_axes"]
    col = [img["hazard_axes"].index(a) for a in axes]
    H = img["H"][:, col]
    X = empirical_cdf_normalize(H)

    out_dir = args.out_dir or (args.ensemble_dir / "diagnostics")
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- numeric diagnostics ------------------------------------------------
    report = dg.coverage_report(H, sel, n_boot=args.n_boot, seed=args.seed)
    gaps = dg.marginal_gaps(H, sel)
    screen = dg.redundancy_screen(H, axes)

    summary = {
        "ensemble_dir": str(args.ensemble_dir),
        "n_pool": int(H.shape[0]),
        "n_selected": report.n,
        "hazard_axes": axes,
        "coverage": {
            "l2_star": report.l2_star,
            "random_l2_mean": report.random_l2_mean,
            "random_l2_std": report.random_l2_std,
            "discrepancy_percentile": report.discrepancy_percentile,
            "nn_min": report.nn_min,
            "nn_mean": report.nn_mean,
            "nn_cv": report.nn_cv,
            "mst_mean": report.mst_mean,
        },
        "marginal_max_gap": {axes[i]: float(gaps[i]) for i in range(len(axes))},
        "redundant_axis_pairs": screen["redundant_pairs"],
    }
    (out_dir / "diagnostics_summary.json").write_text(json.dumps(summary, indent=2))

    print(f"[diag] pool={summary['n_pool']} selected={report.n} axes={axes}")
    print(
        f"[diag] L2-star selected={report.l2_star:.4f} vs random "
        f"{report.random_l2_mean:.4f} +/- {report.random_l2_std:.4f} "
        f"(beats {(1 - report.discrepancy_percentile) * 100:.0f}% of random draws)"
    )
    print(f"[diag] nn_min={report.nn_min:.4f} nn_cv={report.nn_cv:.3f} "
          f"mst_mean={report.mst_mean:.4f}")
    print(f"[diag] marginal max gaps: "
          f"{ {axes[i]: round(float(gaps[i]), 3) for i in range(len(axes))} }")
    if screen["redundant_pairs"]:
        print(f"[diag] REDUNDANT axis pairs (|rho_S|>=0.7): {screen['redundant_pairs']}")
    else:
        print("[diag] no redundant axis pairs (|rho_S|>=0.7).")

    # --- figures ------------------------------------------------------------
    plot_hazard_pair_grid(X, sel, axes, out_dir / "hazard_pair_grid.png")
    plot_hazard_pair_grid_raw(H, sel, axes, out_dir / "hazard_pair_grid_raw.png")
    plot_marginal_coverage(X, sel, axes, gaps, out_dir / "marginal_coverage.png")
    plot_discrepancy_in_context(report, out_dir / "discrepancy_in_context.png", seed=args.seed)
    print(f"[diag] wrote summary + 4 figures -> {out_dir}")


if __name__ == "__main__":
    main()
