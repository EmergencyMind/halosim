"""
training.py — Training program simulation.

Non-targeted programs (monthly, bi-monthly, quarterly, custom):
  Vectorized — training events applied to all on-shift providers on scheduled days.

Targeted program (Advanced only):
  Sequential day-loop — each day, compute current readiness and train only
  providers below the threshold who are currently on shift.
  Cannot be vectorized because each day's decision depends on prior state.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def compute_training(sim) -> np.ndarray:
    """
    Return training_matrix: (n_providers, n_days) bool.
    True on days when a provider receives a training event.
    """
    program = sim.training_program

    if program == "none":
        return np.zeros((sim.n_providers, sim.n_days), dtype=bool)

    elif program in ("monthly", "bimonthly", "quarterly"):
        intervals = {"monthly": 28, "bimonthly": 56, "quarterly": 84}
        return _scheduled_training(
            sim,
            interval=intervals[program],
            start=sim.training_start_day,
        )

    elif program == "custom":
        return _scheduled_training(
            sim,
            interval=sim.training_interval_days,
            start=sim.training_start_day,
        )

    elif program == "targeted":
        return _targeted_training(sim)

    return np.zeros((sim.n_providers, sim.n_days), dtype=bool)


# ---------------------------------------------------------------------------
# Non-targeted (vectorized)
# ---------------------------------------------------------------------------

def _scheduled_training(sim, interval: int, start: int = 0) -> np.ndarray:
    """
    All on-shift providers receive training on each scheduled day
    (start, start+interval, start+2*interval, ...).
    """
    training = np.zeros((sim.n_providers, sim.n_days), dtype=bool)
    on_shift = sim.on_shift_mask   # (n_providers, n_days)

    training_days = np.arange(start, sim.n_days, interval)
    for t in training_days:
        if 0 <= t < sim.n_days:
            training[:, t] = on_shift[:, t]

    return training


# ---------------------------------------------------------------------------
# Targeted (sequential day-loop)
# ---------------------------------------------------------------------------

def _targeted_training(sim) -> np.ndarray:
    """
    Each day: compute readiness for on-shift providers; train those below
    sim.training_target_threshold.  Training budget: once per interval.

    This is an agent-based simulation — cannot be vectorized.
    """
    from halosim.readiness import _days_since_last_reset

    n_providers = sim.n_providers
    n_days = sim.n_days
    threshold = sim.training_target_threshold
    interval = sim.training_interval_days
    start = sim.training_start_day
    never = n_days + 1

    training = np.zeros((n_providers, n_days), dtype=bool)
    last_reset = np.full(n_providers, -never, dtype=np.int32)  # "never trained/exposed"
    next_training_day = start                                   # budget tracker

    exposure_matrix = sim.exposure_matrix
    on_shift = sim.on_shift_mask

    model = sim.readiness_model
    r_threshold = sim.readiness_threshold_days
    half_life = sim.readiness_half_life_days
    b = sim.ebbinghaus_b
    t1 = sim.readiness_threshold_days
    t2 = sim.step_t2_days
    partial = sim.step_partial_value
    effect = sim.training_equivalence  # partial boost factor

    def readiness_fn(days_since: np.ndarray) -> np.ndarray:
        if model == "binary":
            return (days_since <= r_threshold).astype(float)
        elif model == "exponential":
            lam = np.log(2.0) / max(half_life, 1.0)
            return np.exp(-lam * days_since)
        elif model == "ebbinghaus":
            return 1.0 / (1.0 + b * days_since)
        elif model == "step":
            r = np.zeros_like(days_since, dtype=float)
            r[days_since <= t1] = 1.0
            r[(days_since > t1) & (days_since <= t2)] = partial
            return r
        return (days_since <= r_threshold).astype(float)

    for t in range(n_days):
        # Update last_reset from live exposures today
        exposed_today = exposure_matrix[:, t]
        last_reset[exposed_today] = t

        # Compute readiness at start of today
        days_since = np.clip(t - last_reset, 0, never).astype(np.int32)
        readiness_today = readiness_fn(days_since.astype(float))

        on_today = on_shift[:, t]

        # Only train if it's a scheduled training day (budget)
        if t >= next_training_day:
            needs_training = on_today & (readiness_today < threshold)
            if needs_training.any():
                training[needs_training, t] = True
                # Partial vs full reset
                if effect < 1.0:
                    # Partial: shift last_reset forward proportionally
                    effective_days_back = int((1.0 - effect) * days_since[needs_training].mean())
                    last_reset[needs_training] = max(0, t - effective_days_back)
                else:
                    last_reset[needs_training] = t
            next_training_day = t + interval

    return training
