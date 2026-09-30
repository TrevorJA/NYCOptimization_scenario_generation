"""Forcing space: hybrid CMIP6 envelope + hypercube fill (LHS or i.i.d.; methods 3.1).

Builds the deeply-uncertain forcing-parameter space ``theta`` (12-month multiplicative change-factor
profiles) and applies it externally to a fitted Kirsch generator via the log-space transform
(Kirsch et al. 2013, eqs. 10-11). The same forcing space drives the deeply-uncertain / resampled
generation and the test ensemble.

Two CMIP6-derived axes are supported (the input-vs-hazard coverage diagnostic uses both):

  - **Mean change ``a_j``** -- the *multiplicative* ``_frac_`` twin CSV (already a_j; baseline
    column == 1.0); see :func:`load_cmip6_envelope`::

        ../CMIP6_multimodel_streamflow/stats/diff_relative_to_dataset_baseline/
            nyc_inflow_monthly_mean_frac_by_dataset_ssp_and_period.csv

  - **CV (variance) change ``v_j``** -- derived from the absolute monthly mean/std tables by
    sibling-matching each future GCM run to its own historical (``1980_2019``) period; see
    :func:`derive_variance_envelope`. ``v_j`` is an *independent* perturbation on top of the
    CV-preserving baseline (``c_j = a_j``): ``c_j = a_j * v_j``.

The CSV month index is CALENDAR (1=Jan); Kirsch/Pywr-DRB are water-year aligned (Oct start) -> the
profiles MUST be re-indexed calendar->water-year before they are applied to ``mean_period``/``std_period``.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Final, Mapping

import numpy as np
import pandas as pd

from .subsample import generate_lhs_samples

#: Default location of the multiplicative CMIP6 monthly change-factor table (a_j), relative to the
#: sibling CMIP6 repo. Resolve against the actual checkout at call time.
CMIP6_FRAC_CSV: Final[str] = (
    "CMIP6_multimodel_streamflow/stats/diff_relative_to_dataset_baseline/"
    "nyc_inflow_monthly_mean_frac_by_dataset_ssp_and_period.csv"
)

#: Water-year month order (Oct..Sep) used by the Kirsch ``period`` index.
WATER_YEAR_MONTHS: Final[tuple[int, ...]] = (10, 11, 12, 1, 2, 3, 4, 5, 6, 7, 8, 9)

#: Fundamental angular frequency of the 12-month seasonal cycle (rad/month).
_OMEGA: Final[float] = 2.0 * np.pi / 12.0

#: Matches a CMIP6 GCM/SSP run column (``..._ssp245_...``), excluding pure-historical datasets.
_SSP_RE: Final[re.Pattern] = re.compile(r"ssp\d", re.IGNORECASE)


def _ssp_columns(df: pd.DataFrame) -> list[str]:
    """CMIP6 GCM/SSP anchor columns (those carrying an ``sspNNN`` token)."""
    return [c for c in df.columns if _SSP_RE.search(c)]


def _gcm_key(col: str) -> str:
    """Strip the SSP/period suffix to a ``{hydro}_RAPID_{GCM}`` sibling key."""
    return re.sub(r"_ssp\d+_.*", "", col, flags=re.IGNORECASE)


def _read_month_indexed(csv_path: str | Path) -> pd.DataFrame:
    """Read a stats CSV indexed by calendar ``month`` (1..12)."""
    df = pd.read_csv(csv_path).set_index("month")
    df.index = df.index.astype(int)
    return df


def _to_water_year(df_cal: pd.DataFrame) -> pd.DataFrame:
    """Reorder a calendar-month-indexed (1..12) frame to water-year order (Oct..Sep)."""
    return df_cal.loc[list(WATER_YEAR_MONTHS)]


def load_cmip6_envelope(csv_path: str | Path, *, include_baseline: bool = False) -> pd.DataFrame:
    """Load the multiplicative CMIP6 mean change-factor profiles ``a_j``.

    Args:
        csv_path: Path to the ``_frac_`` CSV (12 calendar-month rows x scenario columns).
        include_baseline: If False (default) drop the historical-period (``1980_2019``) anchors and
            keep only the *future* projections, so the envelope spans projected change rather than
            current climate.

    Returns:
        DataFrame of multiplicative factors ``a_j`` indexed by water-year month (Oct..Sep), one
        column per CMIP6 future scenario.
    """
    df = _read_month_indexed(csv_path)
    cols = _ssp_columns(df)
    if not include_baseline:
        cols = [c for c in cols if "1980_2019" not in c]
    if not cols:
        raise ValueError(f"no CMIP6 (ssp) anchor columns found in {csv_path}")
    out = _to_water_year(df.loc[:, cols].astype(float))
    if not (0.2 < float(out.values.mean()) < 5.0):
        raise ValueError(
            f"{csv_path} does not look like the multiplicative `_frac_` table "
            f"(mean={out.values.mean():.3g}); pass the _frac_ variant, not _prc_change_."
        )
    return out


def derive_variance_envelope(
    mean_csv: str | Path,
    std_csv: str | Path,
    *,
    baseline_token: str = "1980_2019",
) -> pd.DataFrame:
    """Derive CMIP6 CV-change factors ``v_j`` from absolute monthly mean/std tables.

    For each future GCM/SSP run, the CV change relative to that run's own historical period
    (``baseline_token``) is ``v_j = (std_f / std_b) / (mean_f / mean_b)``. ``v_j = 1`` means the
    coefficient of variation is unchanged (the CV-preserving baseline); ``v_j != 1`` is the
    independent variance perturbation applied as ``c_j = a_j * v_j``.

    Args:
        mean_csv: Absolute monthly-mean table (``datasets_nyc_inflow_monthly_means.csv``).
        std_csv: Absolute monthly-std table (``datasets_nyc_inflow_monthly_stds.csv``).
        baseline_token: Period substring identifying each GCM's historical sibling.

    Returns:
        DataFrame of CV-change factors ``v_j`` indexed by water-year month (Oct..Sep), one column
        per matched future scenario.
    """
    mean = _read_month_indexed(mean_csv)
    std = _read_month_indexed(std_csv)
    ssp = [c for c in _ssp_columns(mean) if c in std.columns]
    basemap = {_gcm_key(c): c for c in ssp if baseline_token in c}
    futures = [c for c in ssp if baseline_token not in c]

    cv: dict[str, pd.Series] = {}
    for c in futures:
        b = basemap.get(_gcm_key(c))
        if b is None:
            continue
        cv[c] = (std[c] / std[b]) / (mean[c] / mean[b])
    if not cv:
        raise ValueError(
            f"no future/baseline sibling pairs found (baseline_token={baseline_token!r})"
        )
    return _to_water_year(pd.DataFrame(cv).astype(float))


def calendar_to_water_year(profile: np.ndarray) -> np.ndarray:
    """Reorder a 12-vector from calendar-month (Jan..Dec) to water-year (Oct..Sep) order."""
    profile = np.asarray(profile)
    if profile.shape[-1] != 12:
        raise ValueError("profile last axis must be length 12 (months)")
    idx = [m - 1 for m in WATER_YEAR_MONTHS]
    return profile[..., idx]


def water_year_to_calendar(profile: np.ndarray) -> np.ndarray:
    """Reorder a 12-vector from water-year (Oct..Sep) back to calendar-month (Jan..Dec) order."""
    profile = np.asarray(profile)
    if profile.shape[-1] != 12:
        raise ValueError("profile last axis must be length 12 (months)")
    inv = [WATER_YEAR_MONTHS.index(m) for m in range(1, 13)]
    return profile[..., inv]


def fit_harmonic_params(envelope: pd.DataFrame | np.ndarray, *, order: int = 2) -> dict:
    """Fit a low-order harmonic (Fourier) model to each anchor's log change-factor profile.

    For each anchor column of ``envelope`` (a water-year 12-month change-factor profile), least-squares
    fit ``ln a(t) = m + sum_h r_h cos(h*omega*t - psi_h)`` with ``omega = 2*pi/12``. On 12 equally
    spaced points the basis is orthogonal (a truncated DFT), so the fit is exact for a band-limited
    profile and the coefficients are interpretable: ``m`` = log annual-mean level, ``r_h`` / ``psi_h``
    = amplitude / phase of harmonic ``h`` (h=1 annual, h=2 semiannual).

    Args:
        envelope: ``(12, K)`` water-year anchor profiles (columns are anchors).
        order: Number of harmonics to retain.

    Returns:
        Dict with ``m`` ``(K,)``, ``amp`` ``(K, order)``, ``phase`` ``(K, order)`` (radians), ``order``.
    """
    A = np.asarray(envelope, dtype=float)
    if A.shape[0] != 12:
        raise ValueError(f"envelope must have 12 rows (water-year months); got {A.shape}")
    L = np.log(A.T)  # (K, 12)
    t = np.arange(12)
    cols = [np.ones(12)]
    for h in range(1, order + 1):
        cols += [np.cos(h * _OMEGA * t), np.sin(h * _OMEGA * t)]
    X = np.column_stack(cols)
    beta, *_ = np.linalg.lstsq(X, L.T, rcond=None)  # (1 + 2*order, K)
    amp = np.empty((A.shape[1], order))
    phase = np.empty((A.shape[1], order))
    for h in range(order):
        a_h, b_h = beta[1 + 2 * h], beta[2 + 2 * h]
        amp[:, h] = np.hypot(a_h, b_h)
        phase[:, h] = np.arctan2(b_h, a_h)
    return {"m": beta[0], "amp": amp, "phase": phase, "order": int(order)}


def reconstruct_harmonic(params: np.ndarray, *, order: int = 2, floor: float = 0.2) -> np.ndarray:
    """Reconstruct water-year change-factor profiles from harmonic parameter vectors.

    Args:
        params: ``(n, 1 + 2*order)`` rows ordered ``[m, r1, psi1, r2, psi2, ...]`` (phases in radians).
        order: Number of harmonics (must match ``params`` width).
        floor: Lower clip on the returned multipliers.

    Returns:
        ``(n, 12)`` water-year multiplicative change factors ``exp(ln a(t))``.
    """
    P = np.atleast_2d(np.asarray(params, dtype=float))
    t = np.arange(12)
    L = P[:, [0]] * np.ones(12)[None, :]
    for h in range(order):
        r = P[:, 1 + 2 * h][:, None]
        psi = P[:, 2 + 2 * h][:, None]
        L = L + r * np.cos((h + 1) * _OMEGA * t[None, :] - psi)
    return np.clip(np.exp(L), floor, None)


def harmonic_param_box(
    fit: dict, *, bound_pct: tuple[float, float] = (5.0, 95.0), margin: float = 0.0
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Per-parameter hypercube bounds from the empirical CMIP6 parameter range.

    Bounds are the per-parameter ``[bound_pct[0], bound_pct[1]]`` percentiles across anchors — the
    default ``(5, 95)`` is the **empirical 90% range** (trim the most extreme tails so single outlier
    GCM runs do not drive the box) — optionally widened by ``margin`` * range; amplitude lower bounds
    are floored at 0. The box is sampled independently per axis (deeply-uncertain DMDU exploration, NOT
    the anchors' joint distribution).

    Args:
        fit: Output of :func:`fit_harmonic_params`.
        bound_pct: Lower/upper percentiles defining each per-parameter range (default 90% range).
        margin: Optional fractional widening of each range (negative trims further).

    Returns:
        ``(lo, hi, names)`` with names ordered ``[m, r1, psi1, r2, psi2, ...]``.
    """
    order = fit["order"]
    arrs = [np.asarray(fit["m"], dtype=float)]
    names = ["m"]
    for h in range(order):
        arrs += [fit["amp"][:, h], fit["phase"][:, h]]
        names += [f"r{h + 1}", f"psi{h + 1}"]
    lo_p, hi_p = bound_pct
    lo, hi = [], []
    for name, a in zip(names, arrs):
        low, high = float(np.percentile(a, lo_p)), float(np.percentile(a, hi_p))
        rng = high - low if high > low else abs(high) + 1e-9
        low, high = low - margin * rng, high + margin * rng
        if name.startswith("r"):
            low = max(0.0, low)
        lo.append(low)
        hi.append(high)
    return np.array(lo), np.array(hi), names


def canonical_phases(envelope: pd.DataFrame | np.ndarray, *, order: int = 2) -> np.ndarray:
    """Phases of the harmonic fit to the CMIP6 ensemble-mean log change profile.

    These define the *canonical* CMIP6 seasonal shape (peak month and rise/shoulder asymmetry). The
    individual-anchor phases are noisy — especially the semiannual phase, which is ill-determined when
    its amplitude is small — so fixing the phases at this ensemble-mean fit anchors every sampled
    profile to the characteristic CMIP6 shape (cf. Quinn et al. 2018, who hold baseline harmonic phases
    fixed and perturb only the amplitudes).

    Returns:
        Length-``order`` array of phases (radians).
    """
    A = np.asarray(envelope, dtype=float)
    mean_profile = np.exp(np.log(A.T).mean(axis=0))  # geometric-mean change factor, (12,)
    return fit_harmonic_params(mean_profile[:, None], order=order)["phase"][0]


def _draw_box(
    n: int, lo: np.ndarray, hi: np.ndarray, *, seed: int, method: str
) -> np.ndarray:
    """Draw ``n`` points from the hyper-rectangle ``[lo, hi]`` by LHS or i.i.d. Monte Carlo.

    Args:
        n: Number of points.
        lo: ``(d,)`` per-axis lower bounds.
        hi: ``(d,)`` per-axis upper bounds.
        seed: RNG seed.
        method: ``"lhs"`` (stratified) or ``"iid"`` (plain uniform Monte Carlo).

    Returns:
        ``(n, d)`` sample matrix.

    Raises:
        ValueError: If ``method`` is not ``"lhs"`` or ``"iid"``.
    """
    lo = np.asarray(lo, dtype=float)
    hi = np.asarray(hi, dtype=float)
    if method == "lhs":
        return generate_lhs_samples(n, len(lo), lo, hi, seed=seed)
    if method == "iid":
        return np.random.default_rng(seed).uniform(lo, hi, size=(n, len(lo)))
    raise ValueError(f"unknown sampling method: {method!r}")


def override_axis_bounds(
    lo: np.ndarray, hi: np.ndarray, names: list[str],
    axis_bounds: Mapping[str, tuple[float | None, float | None]],
) -> tuple[np.ndarray, np.ndarray]:
    """Replace named per-axis bounds of a harmonic-parameter box.

    Args:
        lo: ``(d,)`` lower bounds of the box (from :func:`harmonic_param_box`).
        hi: ``(d,)`` upper bounds.
        names: Axis names in box order.
        axis_bounds: ``{axis: (lo, hi)}``; ``None`` on either side keeps the box value.
            Amplitude lower bounds stay floored at 0.

    Returns:
        ``(lo, hi)`` copies with the overrides applied.

    Raises:
        ValueError: On an unknown axis name or an empty interval.
    """
    lo, hi = np.array(lo, dtype=float), np.array(hi, dtype=float)
    for name, (new_lo, new_hi) in axis_bounds.items():
        if name not in names:
            raise ValueError(f"axis_bounds names unknown axis {name!r}; box axes are {names}")
        i = names.index(name)
        if new_lo is not None:
            lo[i] = max(0.0, float(new_lo)) if name.startswith("r") else float(new_lo)
        if new_hi is not None:
            hi[i] = float(new_hi)
        if not lo[i] < hi[i]:
            raise ValueError(f"axis_bounds for {name!r} give an empty interval [{lo[i]}, {hi[i]}]")
    return lo, hi


def sample_harmonic_forcing(
    n_profiles: int,
    envelope: pd.DataFrame | np.ndarray,
    *,
    seed: int,
    bound_pct: tuple[float, float] = (5.0, 95.0),
    margin: float = 0.0,
    order: int = 2,
    floor: float = 0.2,
    fix_phase: bool = True,
    return_params: bool = False,
    method: str = "lhs",
    axis_bounds: Mapping[str, tuple[float | None, float | None]] | None = None,
):
    """Draw forcing profiles over the CMIP6-based interpretable harmonic-parameter hypercube.

    Fits the harmonic model to the CMIP6 anchors (:func:`fit_harmonic_params`), forms the parameter box
    (:func:`harmonic_param_box`), samples it independently per axis (deeply-uncertain exploration), and
    reconstructs the profiles (:func:`reconstruct_harmonic`). The canonical forcing-space sampler.

    With ``fix_phase=True`` (default, the Quinn et al. 2018 structure) the harmonic **phases are held at
    the canonical CMIP6 shape** (:func:`canonical_phases`) and only the **amplitudes** ``[m, r1, r2,
    ...]`` are sampled — so every profile retains the characteristic CMIP6 seasonal shape (correct peak
    month and asymmetry), scaled. This both improves shape fidelity and curbs the over-dispersion that
    independent phase sampling causes. With ``fix_phase=False`` all
    ``[m, r1, psi1, r2, psi2, ...]`` are sampled (legacy behavior).

    Args:
        n_profiles: Number of profiles to draw.
        envelope: ``(12, K)`` water-year CMIP6 anchor profiles.
        seed: Sampler RNG seed.
        bound_pct: Percentiles defining each parameter's CMIP6 range (default (5, 95) = 90% range).
        margin: Optional fractional widening of the parameter box.
        order: Number of harmonics.
        floor: Positivity floor on the returned multipliers.
        fix_phase: If True, fix phases at the canonical CMIP6 shape and sample only amplitudes.
        return_params: If True, also return the sampled parameter matrix and axis names (amplitudes
            only when ``fix_phase``).
        method: How the parameter box is sampled — ``"lhs"`` (Latin hypercube, stratified per axis)
            or ``"iid"`` (independent uniform draws over the box).

            **``"iid"`` exists to keep the scenario-design comparison a valid control, and must not be
            silently replaced by LHS.** The control argument is a distributional-equivalence one: a
            uniform random size-``N`` subset of an **i.i.d.** pool is distributionally identical to
            ``N`` i.i.d. draws, so a design that generates ``N`` i.i.d. realizations is the *exact*
            control for a design that space-filling-subsamples a pool of i.i.d. realizations. That
            equivalence FAILS for an LHS pool: a random subset of an LHS design is not i.i.d. (LHS
            imposes per-axis stratification that a subset inherits only partially). Candidate pools
            that are later subsampled — and the probabilistic designs whose reference measure is
            plain Monte Carlo over the deeply-uncertain hypercube — therefore require ``"iid"``.
            ``"lhs"`` remains correct where the *drawn sample itself* is used as a space-filling
            design and is never subsampled.

        axis_bounds: Optional per-axis overrides of the box, ``{axis: (lo, hi)}`` on the
            intrinsic names (``m``, ``r1``, ``r2``, ...); ``None`` keeps the box value. Applied
            after ``bound_pct`` and ``margin`` (:func:`override_axis_bounds`), so one axis can
            extend beyond the CMIP6 span asymmetrically, which is how the held-out test
            ensemble sets its annual-volume lower bound.

    Returns:
        ``(n_profiles, 12)`` water-year change factors; or ``(profiles, params, names)`` if
        ``return_params``, where ``params`` is the ``(n_profiles, d)`` matrix of **intrinsic**
        harmonic coordinates and ``names`` its axis labels: ``[m, r1, r2, ...]`` when ``fix_phase``
        (phases are fixed, hence not free coordinates), else ``[m, r1, psi1, r2, psi2, ...]``.

    Raises:
        ValueError: If ``method`` is not ``"lhs"`` or ``"iid"``.
    """
    fit = fit_harmonic_params(envelope, order=order)
    lo, hi, names = harmonic_param_box(fit, bound_pct=bound_pct, margin=margin)
    if axis_bounds:
        lo, hi = override_axis_bounds(lo, hi, names, axis_bounds)
    if not fix_phase:
        plan = _draw_box(n_profiles, lo, hi, seed=seed, method=method)
        profiles = reconstruct_harmonic(plan, order=order, floor=floor)
        return (profiles, plan, names) if return_params else profiles

    # Quinn-style: sample amplitudes only [m, r1, r2, ...]; phases fixed at the canonical shape.
    amp_idx = [0] + [1 + 2 * h for h in range(order)]
    amp_names = [names[i] for i in amp_idx]
    psi_c = canonical_phases(envelope, order=order)
    plan_amp = _draw_box(n_profiles, lo[amp_idx], hi[amp_idx], seed=seed, method=method)
    full = np.empty((n_profiles, 1 + 2 * order))
    full[:, 0] = plan_amp[:, 0]
    for h in range(order):
        full[:, 1 + 2 * h] = plan_amp[:, 1 + h]
        full[:, 2 + 2 * h] = psi_c[h]
    profiles = reconstruct_harmonic(full, order=order, floor=floor)
    return (profiles, plan_amp, amp_names) if return_params else profiles


def apply_climate_adjustment(
    mean_period: np.ndarray,
    std_period: np.ndarray,
    a_profile: np.ndarray,
    *,
    c_profile: np.ndarray | None = None,
    preserve_absolute_sd: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    r"""Apply a multiplicative monthly change profile to log-space Kirsch statistics.

    Implements Kirsch et al. (2013) eqs. 10-11. With baseline log-space ``(Y_bar_j, sigma_j)``,
    real-space mean factor ``a_j``, and real-space SD factor ``c_j``:

        arg       = (c/a)^2 (exp(sigma^2) - 1) + 1
        Y_bar_new = ln(a) + Y_bar + sigma^2/2 - 0.5 * ln(arg)
        sigma_new = sqrt( ln(arg) )

    The SD factor ``c_j`` selects the variability convention:

      - ``c_j = a_j`` (CV-preserving) -- the default; the relative drought/flood tail is invariant
        under mean shifts. This is the baseline for a hazard study.
      - ``c_j = 1`` (``preserve_absolute_sd=True``) -- absolute real-space SD held fixed; CV falls
        for wettening and rises for drying (the deliberately-impoverished mean-only convention).
      - ``c_j = a_j * v_j`` -- pass ``c_profile`` directly to add an independent variance
        perturbation ``v_j`` (sampled from the CV envelope by :func:`sample_harmonic_forcing`).

    Args:
        mean_period: Baseline log-space per-month means ``Y_bar_j`` (water-year ordered); shape
            ``(12,)`` or ``(12, n_sites)``.
        std_period: Baseline log-space per-month SDs ``sigma_j``; same shape as ``mean_period``.
        a_profile: Multiplicative monthly mean factors ``a_j`` (water-year ordered), length 12.
        c_profile: Optional real-space SD factors ``c_j`` (length 12). Overrides
            ``preserve_absolute_sd`` when given.
        preserve_absolute_sd: If True and ``c_profile is None``, use ``c_j = 1``; otherwise the
            default is the CV-preserving ``c_j = a_j``.

    Returns:
        ``(mean_period_new, std_period_new)`` to assign onto the fitted generator before generation.
    """
    a = np.asarray(a_profile, dtype=float)
    if c_profile is not None:
        c = np.asarray(c_profile, dtype=float)
    elif preserve_absolute_sd:
        c = np.ones_like(a)
    else:
        c = a.copy()  # CV-preserving
    y_bar = np.asarray(mean_period, dtype=float)
    sigma = np.asarray(std_period, dtype=float)
    if y_bar.ndim == 2:  # broadcast per-month factors across the site axis
        a, c = a[:, None], c[:, None]
    arg = (c / a) ** 2 * (np.exp(sigma**2) - 1.0) + 1.0
    y_bar_new = np.log(a) + y_bar + sigma**2 / 2.0 - 0.5 * np.log(arg)
    sigma_new = np.sqrt(np.log(arg))
    return y_bar_new, sigma_new


def forcing_hash(
    profiles: np.ndarray,
    *,
    envelope_csv: str | Path,
    margin: float,
    seed: int,
    start_date: str,
    baseline_period: tuple[str, str],
    full_period: tuple[str, str],
    axis_bounds: Mapping[str, tuple[float | None, float | None]] | None = None,
) -> str:
    """Return a stable content hash of the forcing-space configuration for the provenance manifest.

    Covers the profiles, the envelope config, the realization stamp epoch, and the
    generator fit-record bounds — everything whose change alters the staged flows
    for a fixed seed. Uses ``hashlib.sha1`` (NOT the process-salted builtin ``hash``).
    """
    h = hashlib.sha1()
    h.update(np.ascontiguousarray(np.asarray(profiles, dtype=float)).tobytes())
    h.update(str(Path(envelope_csv)).encode())
    h.update(f"{margin!r}|{seed!r}".encode())
    h.update(f"{start_date!r}|{tuple(baseline_period)!r}|{tuple(full_period)!r}".encode())
    if axis_bounds:
        h.update(repr(sorted((k, tuple(v)) for k, v in axis_bounds.items())).encode())
    return h.hexdigest()
