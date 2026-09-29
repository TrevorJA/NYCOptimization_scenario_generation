"""Pins each named scoring rule to its columns and scored values. Skipped without SynHydro.

Every hazard image records the rules it was scored under (``dry_scoring_rule``,
``wet_scoring_rule``, ``supplement_scoring_rule``) and readers refuse an image
recording another rule. The guard holds only if a rule is renamed whenever its
scoring changes, so each rule's columns and its values on three fixed scenarios
are pinned here under the rule's name. A failure means images scored before the
change are not commensurable with new ones: give the rule a new name in
``scengen.hazard_metrics``, then re-pin.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("synhydro", reason="hazard_metrics needs SynHydro's SSI")

from scengen import hazard_metrics as hm  # noqa: E402

_SCENARIO_START = "2002-12-01"

#: ``(phase, [(first day, last day, flow factor), ...])`` per scenario. The
#: controlling drought lies inside the window in the first, is still open at
#: the window end in the second and is under way at the window start in the
#: third, so every branch of the phase rates is pinned.
_SCENARIOS = (
    (0.3, [("2008-04-10", "2008-04-14", 8.0)]),
    (1.8, [("2011-05-01", "2011-12-31", 0.5), ("2012-01-01", "2012-05-31", 0.75)]),
    (2.6, [("2002-12-01", "2003-04-30", 0.75), ("2003-05-01", "2003-10-31", 0.5)]),
)

_PINS = {
    "dry": {
        "constant": "_DRY_SCORING_RULE",
        "rule": "gamma2-ssi+open-end-events+in-window-rates",
        "columns": (
            "drought_magnitude", "drought_duration", "drought_severity",
            "drought_development_rate", "drought_termination_rate",
        ),
        "values": [
            [31.42620088, 27, 1.852262218, 0.1157663886, 0.1543551848],
            [29.22486653, 11, 3.675741761, 0.6126236268, 0.1957683928],
            [30.5164107, 29, 3.522347167, 0.4597174962, 0.1408938867],
        ],
    },
    "wet": {
        "constant": "_WET_SCORING_RULE",
        "rule": "q95-pot+critical-pulse+one-day-rise",
        "columns": ("flood_peak_discharge", "flood_pulse_duration", "flood_rise_rate"),
        "values": [
            [13.9256068, 5, 12.15321138],
            [4.430454252, 5, 1.19660621],
            [4.457220128, 4, 1.173311897],
        ],
    },
    "supplement": {
        "constant": "_SUPPLEMENT_SCORING_RULE",
        "rule": "truncation-flags+event-totals+flow-regime",
        "columns": (
            "drought_onset_truncated", "drought_termination_truncated",
            "drought_event_count", "drought_total_deficit",
            "lowflow_min_12month", "lowflow_min_24month", "lowflow_min_year",
            "lowflow_min_7day", "flood_pulse_count", "flood_days_above",
            "flood_max_3day", "flood_pulse_volume", "highflow_max_year",
            "annual_cv", "flashiness",
        ),
        "values": [
            [0, 0, 2, 49.07089459, 0.7585875668, 0.8014125273, 0.7646643318,
             0.2416271357, 70, 177, 10.88627754, 31.96196128, 1.251869506,
             0.1674533286, 0.2546616496],
            [0, 1, 2, 56.18450513, 0.5807317363, 0.8240432967, 0.5958141361,
             0.1646693239, 57, 155, 4.03019916, 5.471837602, 1.189534445,
             0.1996500389, 0.2502508401],
            [1, 0, 3, 59.33610168, 0.7745563473, 0.8489736397, 0.7745563473,
             0.1713037905, 61, 154, 4.087465301, 6.046525043, 1.269064554,
             0.1660976319, 0.2503017637],
        ],
    },
}


def _flow(days: pd.DatetimeIndex, phase: float) -> np.ndarray:
    """Daily flow with seasonal, interannual and day-to-day cycles (no random draws)."""
    t = np.asarray((days - pd.Timestamp("1900-01-01")).days, dtype=float)
    year = 2.0 * np.pi * t / 365.25
    return 1000.0 * np.exp(
        0.5 * np.cos(year - 1.8)
        + 0.2 * np.sin(year / 3.7 + phase) + 0.1 * np.sin(year / 11.3 + 2.0 * phase)
        + 0.6 * np.sin(2.0 * np.pi * t / 9.3 + phase)
        + 0.4 * np.sin(2.0 * np.pi * t / 23.7 + 3.0 * phase)
    )


def _scenario(phase: float, scalings: list) -> tuple[np.ndarray, np.ndarray, int]:
    """Monthly means, daily flows and the six-month daily cut of one ten-year scenario."""
    t0 = pd.Timestamp(_SCENARIO_START)
    days = pd.date_range(t0, t0 + pd.DateOffset(months=6 + 12 * 9), freq="D")[:-1]
    daily = pd.Series(_flow(days, phase), index=days)
    for first, last, factor in scalings:
        daily.loc[first:last] *= factor
    cut = int((days < t0 + pd.DateOffset(months=6)).sum())
    return daily.resample("MS").mean().to_numpy(), daily.to_numpy(), cut


@pytest.fixture(scope="module")
def scored() -> dict:
    """Columns and scored values of the three scenarios, by part of the image."""
    days = pd.date_range(hm._REFERENCE_START, "2022-12-31", freq="D")
    reference = pd.Series(_flow(days, 0.0), index=days)
    rows = [_scenario(phase, scalings) for phase, scalings in _SCENARIOS]
    H, axes, S, names = hm.compute_candidate_hazard_image(
        np.vstack([r[0] for r in rows]), np.vstack([r[1] for r in rows]),
        reference.resample("MS").mean().to_numpy(), reference.to_numpy(),
        wet_exclusion_days=rows[0][2], return_supplement=True,
        scenario_start=_SCENARIO_START,
    )
    n_dry = len(hm.DRY_EVENT_METRICS)
    return {
        "dry": (axes[:n_dry], H[:, :n_dry]),
        "wet": (axes[n_dry:], H[:, n_dry:]),
        "supplement": (names, S),
    }


@pytest.mark.parametrize("part", list(_PINS))
def test_scoring_rule_is_renamed_when_its_scoring_changes(part, scored):
    pin = _PINS[part]
    rule = getattr(hm, pin["constant"])
    assert rule == pin["rule"], (
        f"hazard_metrics.{pin['constant']} is now {rule!r}: re-pin the {part} columns "
        f"and values of this test under the new name."
    )
    stale = (
        f"the {part} scoring changed under rule {rule!r}. Images scored before the "
        f"change are not commensurable with new ones: give "
        f"hazard_metrics.{pin['constant']} a new name, then re-pin this test."
    )
    columns, values = scored[part]
    assert tuple(columns) == pin["columns"], stale
    np.testing.assert_allclose(values, pin["values"], rtol=1e-6, err_msg=stale)
