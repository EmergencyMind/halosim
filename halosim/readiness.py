"""
readiness.py — Readiness decay models.

All models take a Simulation object and return a (n_providers, n_days) float
matrix where 1.0 = fully ready and 0.0 = not ready.

Binary (Basic):
  ready[i,t] = 1 if days since last reset ≤ threshold, else 0

Exponential decay (Advanced):
  r[i,t] = exp(-λ · days_since)   λ = ln(2) / half_life_days

Ebbinghaus forgetting curve (Advanced):
  r[i,t] = 1 / (1 + b · days_since)

Two-threshold step (Advanced):
  r = 1.0  if days ≤ T1
  r = partial_value  if T1 < days ≤ T2
  r = 0.0  if days > T2

"Last reset" is the most recent day where combined_matrix[i, t] is True
(i.e., a live exposure OR a training event occurred).
"""

from __future__ import annotations

import numpy as np


def compute_readiness(sim) -> np.ndarray:
    """Dispatch to the selected readiness model."""
    model = sim.readiness_model
    combined = sim.combined_matrix   # (n_providers, n_days) bool

    days_since = _days_since_last_reset(combined)   # (n_providers, n_days) int

    if model == "binary":
        return _binary(days_since, sim.readiness_threshold_days)
    elif model == "exponential":
        return _exponential(days_since, sim.readiness_half_life_days)
    elif model == "ebbinghaus":
        return _ebbinghaus(days_since, sim.ebbinghaus_b)
    elif model == "step":
        return _step(
            days_since,
            sim.readiness_threshold_days,
            sim.step_t2_days,
            sim.step_partial_value,
        )
    else:
        return _binary(days_since, sim.readiness_threshold_days)


# ---------------------------------------------------------------------------
# days_since_last_reset (vectorized)
# ---------------------------------------------------------------------------

def _days_since_last_reset(combined: np.ndarray) -> np.ndarray:
    """
    For each provider i on each day t, compute how many days have elapsed
    since the last True in combined[i, :t+1].  If no reset has occurred,
    value is set to a large number (n_days + 1).

    Uses a vectorized cumulative approach — no Python loops.
    """
    n_providers, n_days = combined.shape
    never = n_days + 1

    # cumulative index of last True: broadcast trick
    # For each day t, last_reset[i, t] = max index j ≤ t where combined[i,j] is True
    # If no such j exists, use -never so days_since = t + never
    last = np.full((n_providers, n_days), -never, dtype=np.int32)

    for t in range(n_days):
        if t == 0:
            last[:, 0] = np.where(combined[:, 0], 0, -never)
        else:
            reset_today = combined[:, t]
            last[:, t] = np.where(reset_today, t, last[:, t - 1])

    days_since = np.arange(n_days, dtype=np.int32)[np.newaxis, :] - last
    days_since = np.clip(days_since, 0, never)
    return days_since


# ---------------------------------------------------------------------------
# Model implementations
# ---------------------------------------------------------------------------

def _binary(days_since: np.ndarray, threshold: int) -> np.ndarray:
    return (days_since <= threshold).astype(float)


def _exponential(days_since: np.ndarray, half_life: float) -> np.ndarray:
    lam = np.log(2.0) / max(half_life, 1.0)
    return np.exp(-lam * days_since)


def _ebbinghaus(days_since: np.ndarray, b: float) -> np.ndarray:
    return 1.0 / (1.0 + b * days_since)


def _step(
    days_since: np.ndarray,
    t1: int,
    t2: int,
    partial: float,
) -> np.ndarray:
    r = np.zeros_like(days_since, dtype=float)
    r[days_since <= t1] = 1.0
    r[(days_since > t1) & (days_since <= t2)] = partial
    return r
