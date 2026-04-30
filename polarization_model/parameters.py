"""
Simulation configuration and spatial primitives.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class Location:
    """A grid cell used for homes, workplaces, and schools."""

    x: int
    y: int
    location_type: str = "empty"

    def __hash__(self):
        return hash((self.x, self.y))

    def __eq__(self, other):
        if other is None:
            return False
        return self.x == other.x and self.y == other.y


@dataclass
class ModelParameters:
    """
    All tuneable knobs for a single simulation run.

    Adjust these in experiment scripts rather than editing model internals.
    See README for guidance on which parameters drive which outcomes.
    """

    # --- Population ---
    num_people: int = 50
    homes_per_quadrant: int = 3

    # --- Bounded-confidence thresholds ---
    # Opinion gap <= threshold_pos  -> agents converge
    # Opinion gap >= threshold_neg  -> agents diverge (polarise)
    threshold_pos: float = 0.4
    threshold_neg: float = 0.6

    # --- Schelling network segregation ---
    # Agents rewire away from connections whose opinion differs by more than
    # segregation_tolerance, driving echo-chamber formation over time.
    segregation_tolerance: float = 0.3
    rewire_prob: float = 0.1

    # --- Economic stress ---
    increase_cost_of_goods: bool = True
    # Mean monthly CPI-U multiplier over 2021-2023 simulation window (BLS, 2024)
    cost_of_goods_growth_rate: float = 1.0038

    # --- Info-source mix (relative weights for random assignment) ---
    parts_govwebsite: int = 5
    parts_x: int = 3
    parts_reddit: int = 4
    parts_facebook: int = 3

    # --- Agent Zero++ action threshold ---
    # Action fires when A + P + C > tau.
    tau_base: float = 0.6
    # Older agents have a slightly higher threshold (less impulsive).
    tau_age_factor: float = 0.005
    # Economic hardship lowers the threshold.
    tau_economic_factor: float = 0.2

    # --- ACT-R memory decay ---
    # Standard literature value is 0.5; lower values = slower forgetting.
    memory_decay: float = 0.5

    # --- Broadcast cadence ---
    broadcast_interval: int = 10

    # --- Initial social network ---
    initial_connections: int = 5

    # --- Reproducibility ---
    seed: Optional[int] = None
