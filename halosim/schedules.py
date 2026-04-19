"""
schedules.py — Provider schedule generation and upload.

Each provider receives a randomly generated individual schedule that fits
the chosen type. All types produce a (n_providers, n_days) matrix where
every cell is one of {'d', 'n', 'o'} (day shift / night shift / off).

Schedule types
--------------
3/7 Day          3 randomly placed day shifts per 7-day week, rest off
3/7 Night        3 randomly placed night shifts per 7-day week, rest off
4/7 Day          4 randomly placed day shifts per 7-day week, rest off
4/7 Night        4 randomly placed night shifts per 7-day week, rest off
Progressive      3–4 shifts per week, each independently day or night
Random           Each day drawn from empirical d/n/o weights (PMID: 41633464)
"""

from __future__ import annotations

import io

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Schedule type catalogue
# ---------------------------------------------------------------------------

SCHEDULE_TYPES: list[str] = [
    "3/7 Day",
    "3/7 Night",
    "4/7 Day",
    "4/7 Night",
    "Progressive (day & night mix)",
    "Random",
]

DEFAULT_SCHEDULE_TYPE = "3/7 Day"

# Empirical d/n/o weights from the paper (PMID: 41633464)
_DEFAULT_WEIGHTS = {"d": 0.246, "n": 0.230, "o": 0.524}

MAX_PROVIDERS = 5_000
WARN_PROVIDERS = 1_000

REQUIRED_UPLOAD_COLS = {"provider_id", "date", "shift_type"}
VALID_SHIFT_TYPES = {"d", "day", "n", "night", "o", "off"}
_NORM = {"d": "d", "day": "d", "n": "n", "night": "n", "o": "o", "off": "o"}


# ---------------------------------------------------------------------------
# Single-provider schedule generator
# ---------------------------------------------------------------------------

def _generate_one(
    rng: np.random.Generator,
    n_days: int,
    schedule_type: str,
    weights: dict[str, float] | None = None,
) -> np.ndarray:
    """Generate a single provider's schedule array of length n_days.

    If `weights` is provided (keys d/n/o, values summing to 1), each day is
    drawn independently from those weights regardless of schedule_type.
    """
    if weights is not None:
        choices = ["d", "n", "o"]
        probs = np.array([weights.get(c, 0.0) for c in choices], dtype=float)
        probs /= probs.sum()
        return rng.choice(choices, size=n_days, p=probs)

    sched = np.full(n_days, "o", dtype="U1")

    if schedule_type in ("3/7 Day", "3/7 Night", "4/7 Day", "4/7 Night"):
        k = 3 if schedule_type.startswith("3") else 4
        char = "d" if "Day" in schedule_type else "n"
        day = 0
        while day < n_days:
            end = min(day + 7, n_days)
            week_len = end - day
            k_actual = min(k, week_len)
            positions = rng.choice(week_len, size=k_actual, replace=False)
            for p in positions:
                sched[day + p] = char
            day += 7

    elif schedule_type == "Progressive (day & night mix)":
        day = 0
        while day < n_days:
            end = min(day + 7, n_days)
            week_len = end - day
            k = int(rng.integers(3, 5))   # 3 or 4 shifts this week
            k_actual = min(k, week_len)
            positions = rng.choice(week_len, size=k_actual, replace=False)
            for p in positions:
                sched[day + p] = rng.choice(["d", "n"])
            day += 7

    elif schedule_type == "Random":
        choices = ["d", "n", "o"]
        probs = np.array([_DEFAULT_WEIGHTS[c] for c in choices])
        probs /= probs.sum()
        sched = rng.choice(choices, size=n_days, p=probs)

    return sched


# ---------------------------------------------------------------------------
# Population schedule generation
# ---------------------------------------------------------------------------

def generate_schedule(
    n_providers: int,
    n_days: int,
    schedule_type: str = DEFAULT_SCHEDULE_TYPE,
    seed: int = 42,
    weights: dict[str, float] | None = None,
) -> tuple[np.ndarray, list[str]]:
    """
    Generate a (n_providers, n_days) schedule matrix.

    Each provider receives an independently randomised schedule that satisfies
    the constraints of schedule_type.  If `weights` is provided (dict with keys
    'd', 'n', 'o' summing to 1), each provider-day is drawn from those weights
    independently (overrides schedule_type).  Seeding is applied at the
    population level so results are fully reproducible given the same inputs.

    Returns (schedule_array, warnings).
    """
    warnings: list[str] = []
    if n_providers > MAX_PROVIDERS:
        n_providers = MAX_PROVIDERS
        warnings.append(f"Population capped at {MAX_PROVIDERS:,} providers.")
    if n_providers > WARN_PROVIDERS:
        warnings.append(f"{n_providers:,} providers — simulation may take a few seconds.")

    if schedule_type not in SCHEDULE_TYPES:
        warnings.append(
            f"Unknown schedule type '{schedule_type}'; defaulting to '{DEFAULT_SCHEDULE_TYPE}'."
        )
        schedule_type = DEFAULT_SCHEDULE_TYPE

    rng = np.random.default_rng(seed)
    schedule = np.empty((n_providers, n_days), dtype="U1")
    for i in range(n_providers):
        schedule[i] = _generate_one(rng, n_days, schedule_type, weights=weights)

    return schedule, warnings


# ---------------------------------------------------------------------------
# Custom 28-day pattern (Advanced mode)
# ---------------------------------------------------------------------------

def generate_from_pattern(
    n_providers: int,
    n_days: int,
    pattern: str,
    seed: int = 42,
) -> tuple[np.ndarray, list[str]]:
    """
    Generate schedules by tiling a custom 28-char d/n/o pattern across n_days.
    Each provider's schedule is identical (pattern determines the template).
    """
    warnings: list[str] = []
    if n_providers > MAX_PROVIDERS:
        n_providers = MAX_PROVIDERS
        warnings.append(f"Population capped at {MAX_PROVIDERS:,} providers.")

    pattern = (pattern[:28] + "o" * 28)[:28]
    base = np.array(list(pattern), dtype="U1")
    repeats = n_days // 28 + 2
    row = np.tile(base, repeats)[:n_days]

    schedule = np.empty((n_providers, n_days), dtype="U1")
    for i in range(n_providers):
        schedule[i] = row

    return schedule, warnings


# ---------------------------------------------------------------------------
# Upload / validation
# ---------------------------------------------------------------------------

def load_schedule_from_upload(
    file_bytes: bytes,
    filename: str,
    n_days: int,
    start_date: str = "2024-01-01",
) -> tuple[np.ndarray | None, list[str], list[str]]:
    """
    Parse and validate an uploaded schedule file.

    Required columns: provider_id, date (YYYY-MM-DD), shift_type (d/day/n/night/o/off)
    Missing dates for a provider are filled as 'o' (off).

    Returns (schedule_array, provider_ids, errors).
    """
    errors: list[str] = []

    try:
        if filename.lower().endswith((".xlsx", ".xls")):
            df = pd.read_excel(io.BytesIO(file_bytes))
        else:
            df = pd.read_csv(io.BytesIO(file_bytes))
    except Exception as e:
        return None, [], [f"Could not parse file: {e}"]

    df.columns = [c.strip().lower() for c in df.columns]

    missing = REQUIRED_UPLOAD_COLS - set(df.columns)
    if missing:
        return None, [], [
            f"Missing required column(s): {', '.join(sorted(missing))}. "
            "Required: provider_id, date, shift_type"
        ]

    df["shift_type"] = df["shift_type"].astype(str).str.strip().str.lower()
    bad = set(df["shift_type"].unique()) - VALID_SHIFT_TYPES
    if bad:
        return None, [], [
            f"Invalid shift_type value(s): {', '.join(sorted(str(b) for b in bad))}. "
            "Allowed: d, day, n, night, o, off"
        ]
    df["shift_type"] = df["shift_type"].map(_NORM)

    try:
        df["date"] = pd.to_datetime(df["date"]).dt.date
    except Exception:
        return None, [], ["Could not parse 'date' column — expected format YYYY-MM-DD."]

    all_dates = [
        (pd.to_datetime(start_date) + pd.Timedelta(days=i)).date()
        for i in range(n_days)
    ]
    date_to_idx = {d: i for i, d in enumerate(all_dates)}

    providers = sorted(df["provider_id"].astype(str).unique())
    n_providers = len(providers)

    if n_providers > MAX_PROVIDERS:
        providers = providers[:MAX_PROVIDERS]
        n_providers = MAX_PROVIDERS
        errors.append(f"File contains more than {MAX_PROVIDERS:,} providers; truncated.")

    p_idx = {p: i for i, p in enumerate(providers)}
    schedule = np.full((n_providers, n_days), "o", dtype="U1")

    for _, row in df.iterrows():
        pid = str(row["provider_id"])
        if pid not in p_idx:
            continue
        d = row["date"]
        if d not in date_to_idx:
            continue
        schedule[p_idx[pid], date_to_idx[d]] = row["shift_type"]

    n_off_by_default = int((schedule == "o").sum())
    total = n_providers * n_days
    if n_off_by_default / total > 0.1:
        errors.append(
            f"Warning: {n_off_by_default:,} provider-days ({n_off_by_default/total:.0%}) "
            "defaulted to 'off' due to missing rows in the upload."
        )

    return schedule, providers, errors
