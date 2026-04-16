"""
events.py — HALO event generation and upload.

Basic:  Poisson MC with a single rate; uniform day/night split.
Advanced: separate day/night rates; seasonal variation; CSV/Excel upload.
"""

from __future__ import annotations

import io
from typing import Literal

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REQUIRED_UPLOAD_COLS = {"date", "shift_type"}
VALID_SHIFT_TYPES = {"day", "night"}


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def generate_events(
    n_days: int,
    rate: float,
    seed: int = 42,
    *,
    day_rate: float | None = None,
    night_rate: float | None = None,
    seasonal_amplitude: float = 0.0,
    seasonal_phase_days: float = 0.0,
    start_date: str = "2024-01-01",
) -> tuple[pd.DataFrame, list[str]]:
    """
    Generate a Poisson event stream.

    Basic mode:  supply only `rate` (events/day); day/night split is 50/50.
    Advanced:    supply `day_rate` and `night_rate` separately (overrides `rate`).
                 Seasonal variation: multiplier = 1 + A * sin(2π(t + φ) / 365).

    Returns (events_df, warnings).
    events_df columns: day_idx (int), date (date), shift_type ('day'|'night')
    """
    warnings: list[str] = []
    rng = np.random.default_rng(seed)

    dates = pd.date_range(start_date, periods=n_days, freq="D")
    t = np.arange(n_days)

    # Seasonal multiplier
    if seasonal_amplitude > 0:
        multiplier = 1.0 + seasonal_amplitude * np.sin(
            2 * np.pi * (t + seasonal_phase_days) / 365.0
        )
        multiplier = np.clip(multiplier, 0.01, None)
    else:
        multiplier = np.ones(n_days)

    if day_rate is not None and night_rate is not None:
        day_lambda = day_rate * multiplier
        night_lambda = night_rate * multiplier
    else:
        half = rate / 2.0
        day_lambda = half * multiplier
        night_lambda = half * multiplier

    if (day_lambda + night_lambda).mean() > 0.10 * 2:
        warnings.append(
            f"Event rate is high (>{0.10:.0%} of days will have events on average). "
            "Verify this matches your HALO event frequency."
        )

    rows = []
    for d in range(n_days):
        n_day = rng.poisson(day_lambda[d])
        n_night = rng.poisson(night_lambda[d])
        for _ in range(n_day):
            rows.append({"day_idx": d, "date": dates[d].date(), "shift_type": "day"})
        for _ in range(n_night):
            rows.append({"day_idx": d, "date": dates[d].date(), "shift_type": "night"})

    df = pd.DataFrame(rows, columns=["day_idx", "date", "shift_type"])
    return df, warnings


# ---------------------------------------------------------------------------
# Upload / validation
# ---------------------------------------------------------------------------

def load_events_from_upload(
    file_bytes: bytes,
    filename: str,
    n_days: int,
    start_date: str = "2024-01-01",
    allow_hour_col: bool = False,
) -> tuple[pd.DataFrame | None, list[str]]:
    """
    Parse and validate an uploaded events file (CSV or Excel).

    Required columns: date (YYYY-MM-DD), shift_type (day|night)
    Optional column:  hour (int 0–23) — enables complex join in Advanced mode

    Returns (events_df, errors).  events_df is None on validation failure.
    """
    errors: list[str] = []

    try:
        if filename.lower().endswith((".xlsx", ".xls")):
            df = pd.read_excel(io.BytesIO(file_bytes))
        else:
            df = pd.read_csv(io.BytesIO(file_bytes))
    except Exception as e:
        return None, [f"Could not parse file: {e}"]

    df.columns = [c.strip().lower() for c in df.columns]

    missing = REQUIRED_UPLOAD_COLS - set(df.columns)
    if missing:
        return None, [
            f"Missing required column(s): {', '.join(sorted(missing))}. "
            f"Required: date, shift_type"
        ]

    # Validate shift_type values
    bad = set(df["shift_type"].dropna().unique()) - VALID_SHIFT_TYPES
    if bad:
        errors.append(
            f"Invalid shift_type value(s): {', '.join(sorted(str(b) for b in bad))}. "
            "Allowed values: day, night"
        )
        return None, errors

    # Parse dates
    try:
        df["date"] = pd.to_datetime(df["date"]).dt.date
    except Exception:
        return None, ["Could not parse 'date' column — expected format YYYY-MM-DD."]

    start = pd.to_datetime(start_date).date()
    end = pd.to_datetime(start_date) + pd.Timedelta(days=n_days - 1)
    end = end.date()

    out_of_window = df[(df["date"] < start) | (df["date"] > end)]
    if len(out_of_window):
        errors.append(
            f"{len(out_of_window)} event(s) fall outside the simulation window "
            f"({start} to {end}) and will be excluded."
        )
    df = df[(df["date"] >= start) & (df["date"] <= end)].copy()

    df["day_idx"] = (pd.to_datetime(df["date"]) - pd.to_datetime(start_date)).dt.days

    if allow_hour_col and "hour" in df.columns:
        df["hour"] = pd.to_numeric(df["hour"], errors="coerce").fillna(12).astype(int)
        df["hour"] = df["hour"].clip(0, 23)

    result_cols = ["day_idx", "date", "shift_type"]
    if allow_hour_col and "hour" in df.columns:
        result_cols.append("hour")

    return df[result_cols].reset_index(drop=True), errors
