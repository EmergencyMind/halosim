"""
simulation.py — Central Simulation object.

All modules read from and write to this dataclass so there is a single
source of truth for every matrix produced during a run.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Simulation dataclass
# ---------------------------------------------------------------------------

@dataclass
class Simulation:
    # ---- inputs set before run() ----------------------------------------
    n_days: int
    providers: list[str]
    # schedule[i, t] in {'d', 'n', 'o'}  shape (n_providers, n_days)
    schedule: np.ndarray | None = None
    # columns: day_idx (int), shift_type ('day'|'night'), optional: hour (int)
    events: pd.DataFrame | None = None
    seed: int = 42

    # ---- training config (set before run()) ------------------------------
    training_program: Literal["none", "monthly", "bimonthly", "quarterly",
                               "custom", "targeted"] = "none"
    training_interval_days: int = 28          # used by 'custom'
    training_start_day: int = 0
    training_effect: Literal["full", "partial"] = "full"
    training_equivalence: float = 1.0         # partial boost factor (0–1)
    training_target_threshold: float = 0.5    # targeted: train if readiness < this

    # ---- readiness config -----------------------------------------------
    readiness_model: Literal["binary", "exponential", "ebbinghaus", "step"] = "binary"
    readiness_threshold_days: int = 90        # binary / step T1
    readiness_half_life_days: float = 60.0    # exponential
    ebbinghaus_b: float = 0.05               # Ebbinghaus shape
    step_partial_value: float = 0.5           # step model partial readiness
    step_t2_days: int = 180                   # step model T2

    # ---- outputs set by run() -------------------------------------------
    exposure_matrix: np.ndarray | None = None   # (n_providers, n_days) bool
    on_shift_mask: np.ndarray | None = None     # (n_providers, n_days) bool
    training_matrix: np.ndarray | None = None   # (n_providers, n_days) bool
    combined_matrix: np.ndarray | None = None   # exposure OR training
    readiness_matrix: np.ndarray | None = None  # (n_providers, n_days) float
    proportion_ready_on_shift: np.ndarray | None = None  # (n_days,) float
    proportion_ready_all: np.ndarray | None = None       # (n_days,) float
    results_df: pd.DataFrame | None = None      # per-provider gap stats

    # ---- validation errors (list of strings) ----------------------------
    errors: list[str] = field(default_factory=list)

    @property
    def n_providers(self) -> int:
        return len(self.providers)

    def is_ready_to_run(self) -> bool:
        return self.schedule is not None and self.events is not None

    def run(self) -> "Simulation":
        """Execute the full pipeline: exposure → training → readiness → aggregate."""
        from halosim.exposure import compute_exposure
        from halosim.training import compute_training
        from halosim.readiness import compute_readiness, _compute_initial_last

        if not self.is_ready_to_run():
            self.errors.append("Schedule and events must be set before running.")
            return self

        # 1. on-shift mask
        self.on_shift_mask = (self.schedule == 'd') | (self.schedule == 'n')

        # 2. exposure
        self.exposure_matrix, self.results_df = compute_exposure(self)

        # 3. Estimate pre-window exposure state from each provider's empirical
        #    rate to avoid left-censoring bias in the readiness timeseries.
        initial_last = _compute_initial_last(self.exposure_matrix, self.n_days, self.seed)

        # 4. training (targeted program uses initial_last for its own day-loop state)
        self.training_matrix = compute_training(self, initial_last=initial_last)
        self.combined_matrix = self.exposure_matrix | self.training_matrix

        # 5. readiness
        self.readiness_matrix = compute_readiness(self, initial_last=initial_last)

        # 5. aggregate readiness (on-shift providers only — primary metric)
        n_days = self.n_days
        prop_on = np.full(n_days, np.nan)
        prop_all = np.zeros(n_days)

        for t in range(n_days):
            on = self.on_shift_mask[:, t]
            if on.any():
                prop_on[t] = self.readiness_matrix[on, t].mean()
            prop_all[t] = self.readiness_matrix[:, t].mean()

        self.proportion_ready_on_shift = prop_on
        self.proportion_ready_all = prop_all

        return self
