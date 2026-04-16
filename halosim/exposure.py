"""
exposure.py — Vectorized exposure computation engine.

Simple join (Basic):  provider is exposed on day t if they are on shift AND
                      a HALO event occurred on that shift that day.

Complex join (Advanced): event is within ±window_hours of any shift boundary;
                         requires 'hour' column in events (upload only).

Per-provider gap statistics match the methodology in Dworkis 2026.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def compute_exposure(sim) -> tuple[np.ndarray, pd.DataFrame]:
    """
    Build the exposure_matrix and per-provider results_df.

    Returns:
      exposure_matrix: (n_providers, n_days) bool
      results_df: per-provider gap statistics DataFrame
    """
    schedule = sim.schedule
    events = sim.events
    n_providers, n_days = schedule.shape

    exposure_matrix = _simple_join(schedule, events, n_days)

    results = []
    for i in range(n_providers):
        stats = _compute_gaps(exposure_matrix[i], sim.readiness_threshold_days)
        stats["provider_id"] = sim.providers[i]
        results.append(stats)

    results_df = pd.DataFrame(results)
    cols = ["provider_id", "n_events", "t2first",
            "gap_min", "gap_q25", "gap_median", "gap_mean", "gap_q75", "gap_max",
            "max_gap_exceeds_threshold"]
    results_df = results_df[cols]

    return exposure_matrix, results_df


# ---------------------------------------------------------------------------
# Simple join (vectorized)
# ---------------------------------------------------------------------------

def _simple_join(
    schedule: np.ndarray,
    events: pd.DataFrame,
    n_days: int,
) -> np.ndarray:
    """Shift-matched join: day events → day-shift providers; night → night."""
    n_providers = schedule.shape[0]
    exposure = np.zeros((n_providers, n_days), dtype=bool)

    day_idx = events[events["shift_type"] == "day"]["day_idx"].values
    night_idx = events[events["shift_type"] == "night"]["day_idx"].values

    # Clip to valid range
    day_idx = day_idx[(day_idx >= 0) & (day_idx < n_days)]
    night_idx = night_idx[(night_idx >= 0) & (night_idx < n_days)]

    working_day = (schedule == "d")    # (n_providers, n_days)
    working_night = (schedule == "n")

    if len(day_idx):
        exposure[:, day_idx] |= working_day[:, day_idx]
    if len(night_idx):
        exposure[:, night_idx] |= working_night[:, night_idx]

    return exposure


# ---------------------------------------------------------------------------
# Complex join (requires hour column; Advanced mode only)
# ---------------------------------------------------------------------------

def compute_exposure_complex(
    sim,
    window_hours: int = 4,
) -> tuple[np.ndarray, pd.DataFrame]:
    """
    Complex join: a provider is exposed if their shift overlaps with the event
    within ±window_hours of the shift boundary (07:00 / 19:00).

    Requires events DataFrame to have an 'hour' column.
    Falls back to simple join if 'hour' is missing.
    """
    if "hour" not in sim.events.columns:
        return compute_exposure(sim)

    schedule = sim.schedule
    events = sim.events
    n_providers, n_days = schedule.shape
    exposure = np.zeros((n_providers, n_days), dtype=bool)

    day_boundary = 7    # day shift starts 07:00
    night_boundary = 19 # night shift starts 19:00

    for _, ev in events.iterrows():
        d = int(ev["day_idx"])
        if d < 0 or d >= n_days:
            continue
        h = int(ev.get("hour", 12))
        st = ev["shift_type"]

        if st == "day":
            # Day shift providers on this day
            on_day = schedule[:, d] == "d"
            # Also catch night providers if event within window_hours before day boundary
            near_boundary = (h >= night_boundary - window_hours) or (h <= day_boundary + window_hours)
            if near_boundary:
                on_night = schedule[:, d] == "n"
                prev_night = schedule[:, max(0, d - 1)] == "n" if d > 0 else np.zeros(n_providers, bool)
                exposure[:, d] |= on_day | on_night | prev_night
            else:
                exposure[:, d] |= on_day
        else:  # night
            on_night = schedule[:, d] == "n"
            near_boundary = (h >= day_boundary - window_hours) or (h >= night_boundary - window_hours)
            if near_boundary:
                on_day = schedule[:, d] == "d"
                exposure[:, d] |= on_night | on_day
            else:
                exposure[:, d] |= on_night

    results = []
    for i in range(n_providers):
        stats = _compute_gaps(exposure[i], sim.readiness_threshold_days)
        stats["provider_id"] = sim.providers[i]
        results.append(stats)

    results_df = pd.DataFrame(results)
    cols = ["provider_id", "n_events", "t2first",
            "gap_min", "gap_q25", "gap_median", "gap_mean", "gap_q75", "gap_max",
            "max_gap_exceeds_threshold"]
    results_df = results_df[cols]

    return exposure, results_df


# ---------------------------------------------------------------------------
# Per-provider gap statistics
# ---------------------------------------------------------------------------

def _compute_gaps(exposure_row: np.ndarray, threshold: int) -> dict:
    """Compute inter-exposure gap statistics for one provider."""
    exposure_days = np.where(exposure_row)[0]
    n = len(exposure_days)

    if n == 0:
        return {
            "n_events": 0,
            "t2first": None,
            "gap_min": None,
            "gap_q25": None,
            "gap_median": None,
            "gap_mean": None,
            "gap_q75": None,
            "gap_max": None,
            "max_gap_exceeds_threshold": True,
        }

    t2first = int(exposure_days[0])

    if n == 1:
        return {
            "n_events": 1,
            "t2first": t2first,
            "gap_min": None,
            "gap_q25": None,
            "gap_median": None,
            "gap_mean": None,
            "gap_q75": None,
            "gap_max": None,
            "max_gap_exceeds_threshold": True,
        }

    gaps = np.diff(exposure_days).astype(float)
    return {
        "n_events": n,
        "t2first": t2first,
        "gap_min": float(gaps.min()),
        "gap_q25": float(np.percentile(gaps, 25)),
        "gap_median": float(np.median(gaps)),
        "gap_mean": float(gaps.mean()),
        "gap_q75": float(np.percentile(gaps, 75)),
        "gap_max": float(gaps.max()),
        "max_gap_exceeds_threshold": bool(gaps.max() > threshold),
    }
