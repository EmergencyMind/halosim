"""
schedules.py — Provider schedule generation and upload.

Basic:   built-in templates (28-day d/n/o patterns), synthetic population.
Advanced: custom 28-char pattern builder; CSV/Excel upload.

Every provider-day resolves to one of {'d', 'n', 'o'}.
  d = day shift (07:00–18:59)
  n = night shift (19:00–06:59)
  o = off
"""

from __future__ import annotations

import io

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Built-in 28-day templates
# ---------------------------------------------------------------------------

def _t(s: str) -> str:
    """Trim/pad a pattern to exactly 28 d/n/o characters."""
    s = s.replace(" ", "o")
    return (s[:28] + "o" * 28)[:28]


TEMPLATES: dict[str, str] = {
    "3-on Day / 4-off":          _t("ddd" + "o"*4 + "ddd" + "o"*4 + "ddd" + "o"*4 + "ddd" + "o"*4),
    "3-on Night / 4-off":        _t("nnn" + "o"*4 + "nnn" + "o"*4 + "nnn" + "o"*4 + "nnn" + "o"*4),
    "4-on Day / 3-off":          _t("dddd" + "o"*3 + "dddd" + "o"*3 + "dddd" + "o"*3 + "dddd" + "o"*3),
    "4-on Night / 3-off":        _t("nnnn" + "o"*3 + "nnnn" + "o"*3 + "nnnn" + "o"*3 + "nnnn" + "o"*3),
    "Rotating Day→Night":        _t("ddd" + "o" + "nnn" + "o" + "ddd" + "o" + "nnn" + "o" + "ddd" + "o" + "nnn" + "o" + "dd"),
    "Progressive (nurse-style)": _t("dndodndodndodndodndodndodndo"),
}

# Empirical d/n/o weights from the paper (Dworkis 2026)
_DEFAULT_WEIGHTS = {"d": 0.246, "n": 0.230, "o": 0.524}

MAX_PROVIDERS = 5_000
WARN_PROVIDERS = 1_000

REQUIRED_UPLOAD_COLS = {"provider_id", "date", "shift_type"}
VALID_SHIFT_TYPES = {"d", "day", "n", "night", "o", "off"}
_NORM = {"d": "d", "day": "d", "n": "n", "night": "n", "o": "o", "off": "o"}


# ---------------------------------------------------------------------------
# xtender — replicate a 28-day pattern to fill n_days (mirrors R logic)
# ---------------------------------------------------------------------------

def xtender(pattern: str, n_days: int) -> np.ndarray:
    """Repeat a 28-char d/n/o pattern to fill exactly n_days."""
    base = np.array(list(pattern), dtype="U1")
    repeats = n_days // 28 + 2
    extended = np.tile(base, repeats)
    return extended[:n_days]


# ---------------------------------------------------------------------------
# Schedule matrix generation
# ---------------------------------------------------------------------------

def generate_schedule(
    n_providers: int,
    n_days: int,
    templates: list[str],
    seed: int = 42,
    weights: dict[str, float] | None = None,
) -> tuple[np.ndarray, list[str]]:
    """
    Generate a (n_providers, n_days) schedule matrix by randomly assigning
    each provider one of the supplied templates.

    Returns (schedule_array, warnings).
    """
    warnings: list[str] = []
    if n_providers > MAX_PROVIDERS:
        n_providers = MAX_PROVIDERS
        warnings.append(f"Population capped at {MAX_PROVIDERS:,} providers.")
    if n_providers > WARN_PROVIDERS:
        warnings.append(
            f"{n_providers:,} providers — simulation may take a few seconds."
        )

    rng = np.random.default_rng(seed)
    patterns = [TEMPLATES.get(t, t) for t in templates]

    # Normalise patterns
    patterns = [(p[:28] + "o" * 28)[:28] for p in patterns]

    if not patterns:
        patterns = list(TEMPLATES.values())

    # Weight assignment by d/n/o content similarity to empirical weights
    # (simple: assign uniformly among supplied templates)
    chosen = rng.integers(0, len(patterns), size=n_providers)

    schedule = np.empty((n_providers, n_days), dtype="U1")
    for i, idx in enumerate(chosen):
        schedule[i] = xtender(patterns[idx], n_days)

    return schedule, warnings


def generate_synthetic_population(
    n_providers: int,
    n_days: int,
    seed: int = 42,
    weights: dict[str, float] | None = None,
) -> tuple[np.ndarray, list[str]]:
    """
    Generate synthetic schedules using empirical d/n/o weights (gensynprov logic
    from waldo_v2_production_070125.R).  Each provider-day is independently sampled.
    """
    warnings: list[str] = []
    if n_providers > MAX_PROVIDERS:
        n_providers = MAX_PROVIDERS
        warnings.append(f"Population capped at {MAX_PROVIDERS:,} providers.")

    w = weights or _DEFAULT_WEIGHTS
    choices = list(w.keys())
    probs = np.array([w[c] for c in choices])
    probs = probs / probs.sum()

    rng = np.random.default_rng(seed)
    flat = rng.choice(choices, size=n_providers * n_days, p=probs)
    return flat.reshape(n_providers, n_days), warnings


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
    Missing dates for a provider → filled as 'o' (off) with a warning.

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

    # Normalise shift_type
    df["shift_type"] = df["shift_type"].astype(str).str.strip().str.lower()
    bad = set(df["shift_type"].unique()) - VALID_SHIFT_TYPES
    if bad:
        return None, [], [
            f"Invalid shift_type value(s): {', '.join(sorted(str(b) for b in bad))}. "
            "Allowed: d, day, n, night, o, off"
        ]
    df["shift_type"] = df["shift_type"].map(_NORM)

    # Parse dates
    try:
        df["date"] = pd.to_datetime(df["date"]).dt.date
    except Exception:
        return None, [], ["Could not parse 'date' column — expected format YYYY-MM-DD."]

    start = pd.to_datetime(start_date).date()
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

    # Warn about coverage gaps
    n_off_by_default = int((schedule == "o").sum())
    total = n_providers * n_days
    if n_off_by_default / total > 0.1:
        errors.append(
            f"Warning: {n_off_by_default:,} provider-days ({n_off_by_default/total:.0%}) "
            "defaulted to 'off' due to missing rows in the upload."
        )

    return schedule, providers, errors
