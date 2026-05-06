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


def compute_readiness(sim, initial_last: np.ndarray | None = None) -> np.ndarray:
    """Dispatch to the selected readiness model."""
    model = sim.readiness_model
    combined = sim.combined_matrix   # (n_providers, n_days) bool

    days_since = _days_since_last_reset(combined, initial_last=initial_last)

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

def _days_since_last_reset(
    combined: np.ndarray,
    initial_last: np.ndarray | None = None,
) -> np.ndarray:
    """
    For each provider i on each day t, compute how many days have elapsed
    since the last True in combined[i, :t+1].  If no reset has occurred,
    value is set to a large number (n_days + 1).

    initial_last: (n_providers,) int32, optional.
        Days-before-window of each provider's last prior reset, expressed as
        negative offsets (e.g. -30 means last reset was 30 days before day 0).
        Providers with no prior history should be set to -(n_days + 1).
        When None, all providers are assumed to have no prior history (original
        behaviour, which causes left-censoring bias at the start of the window).
    """
    n_providers, n_days = combined.shape
    never = n_days + 1

    last = np.full((n_providers, n_days), -never, dtype=np.int32)

    # Day 0: use initial_last as the carry-in prior state
    init_carry = initial_last if initial_last is not None else np.full(n_providers, -never, dtype=np.int32)
    last[:, 0] = np.where(combined[:, 0], 0, init_carry)

    for t in range(1, n_days):
        last[:, t] = np.where(combined[:, t], t, last[:, t - 1])

    days_since = np.arange(n_days, dtype=np.int32)[np.newaxis, :] - last
    days_since = np.clip(days_since, 0, never)
    return days_since


def _compute_initial_last(
    exposure_matrix: np.ndarray,
    n_days: int,
    seed: int,
) -> np.ndarray:
    """
    Estimate each provider's last-exposure day before the window opens, using
    a geometric draw based on their empirical exposure rate in the window.

    Returns initial_last: (n_providers,) int32.
    Values are negative offsets from day 0 (e.g. -30 = last exposed 30 days
    before the window).  Providers with zero exposures get -(n_days + 1).
    """
    never = -(n_days + 1)
    n_providers = exposure_matrix.shape[0]
    n_events = exposure_matrix.sum(axis=1)          # (n_providers,)
    exp_rate = n_events / n_days                    # events/day per provider

    rng = np.random.default_rng(seed + 9973)        # offset so warmup differs from main sim
    initial_last = np.full(n_providers, never, dtype=np.int32)

    active = np.where(exp_rate > 0)[0]
    for i in active:
        p = float(exp_rate[i])
        # Geometric(p): expected gap = 1/p days; models stationary inter-event spacing
        days_since = int(rng.geometric(p=min(p, 0.999)))
        initial_last[i] = -days_since               # negative = before window

    return initial_last


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
