"""
Agent cognitive architecture: ACT-R memory, Agent Zero++ disposition model,
and the Person dataclass that combines them.

Disposition formula (Epstein's Agent Zero++, extended):
    D = (A + P + C + E*0.2) * opinion_amplifier - tau

where:
    A  = Affect        — emotional arousal
    P  = Probability   — rolling-average threat perception
    C  = Contagion     — fraction of network neighbours in active states
    E  = Memory        — normalised ACT-R activation (our extension)
    tau = threshold    — varies by age and economic stress

Action fires when D > 0.  High affect -> FIGHT; lower affect -> FLIGHT.
"""

from __future__ import annotations

import numpy as np
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List

from polarization_model.info_sources import InfoSourceType
from polarization_model.parameters import Location


class AgentState(Enum):
    """Behavioural states from Agent Zero++."""

    QUIET = "quiet"
    AGITATED = "agitated"
    FLIGHT = "flight"    # fleeing/avoiding
    PROTEST = "Protest"
    RIOT = "Riot"
    MOB = "Mob"


class ACTRMemory:
    """
    Simplified ACT-R base-level learning.

    Each exposure to an info item is time-stamped.  Activation decays
    as a power law of elapsed time; repeated exposure keeps items active.

        A_i = ln( sum_j( (t_now - t_j)^(-d) ) )

    Returns values in roughly [-5, 5]; get_avg_activation normalises to [0, 1].
    """

    def __init__(self, decay: float = 0.5):
        self.decay = decay
        # {info_id: [tick_of_exposure, ...]}
        self.exposures: dict[str, list[int]] = defaultdict(list)

    def add_exposure(self, info_id: str, current_tick: int) -> None:
        self.exposures[info_id].append(current_tick)

    def get_activation(self, info_id: str, current_tick: int) -> float:
        if info_id not in self.exposures or not self.exposures[info_id]:
            return float("-inf")

        total = sum(
            max(1, current_tick - t) ** (-self.decay)
            for t in self.exposures[info_id]
        )
        return np.log(total) if total > 0 else float("-inf")

    def get_avg_activation(self, current_tick: int) -> float:
        """Return mean activation across all stored items, normalised to [0, 1]."""
        if not self.exposures:
            return 0.0

        activations = [
            a for info_id in self.exposures
            if (a := self.get_activation(info_id, current_tick)) > float("-inf")
        ]
        if not activations:
            return 0.0

        avg = np.mean(activations)
        return float(np.clip((avg + 5) / 10, 0, 1))


@dataclass
class Person:
    """
    A single agent combining:
      - Demographics and economic state
      - Bounded-confidence opinion dynamics
      - Agent Zero++ disposition (A + P + C -> action)
      - ACT-R episodic memory
      - Schelling-style social network
    """

    unique_id: int

    # --- Demographics ---
    age: int = 0
    employed: bool = False
    income: float = 0.0
    savings: float = 1000.0
    rent: float = 500.0
    unhoused: bool = False

    # --- Spatial ---
    x: float = 0.0
    y: float = 0.0
    home_location: Optional[Location] = None
    work_location: Optional[Location] = None
    school_location: Optional[Location] = None

    # --- Opinion dynamics (bounded confidence) ---
    opinion: float = 0.0
    gamma: float = 0.0         # learning rate for opinion updates
    threshold_pos: float = 0.4
    threshold_neg: float = 0.6

    # --- Info consumption ---
    info_preference: str = "govwebsite"
    trust_gov: float = 0.7
    trust_x: float = 0.5
    trust_reddit: float = 0.5
    trust_facebook: float = 0.4

    # --- Agent Zero++ state variables ---
    affect: float = 0.0         # A: emotional arousal in [0, 1]
    probability: float = 0.0    # P: perceived threat (rolling mean)
    contagion: float = 0.0      # C: fraction of active network neighbours
    disposition: float = 0.0    # D = (A+P+C+E*0.2)*amplifier - tau
    tau: float = 0.6            # action threshold

    impulse_control: float = 1.0  # lower -> more impulsive (driven by age)

    # Per-agent activation thresholds (Granovetter 1978; drawn in _create_people)
    protest_threshold: float = 0.3
    riot_threshold: float = 0.6

    # --- Behavioural state ---
    state: AgentState = AgentState.QUIET

    # --- Social network ---
    connections: List[int] = field(default_factory=list)

    # --- Memory ---
    memory: Optional[ACTRMemory] = None

    # --- Schelling satisfaction ---
    is_happy: bool = True

    # Rolling window of threat observations for P calculation
    threat_memory: List[float] = field(default_factory=list)
    memory_window: int = 10

    def __post_init__(self):
        if self.memory is None:
            self.memory = ACTRMemory()

    # ------------------------------------------------------------------
    # Trust helpers
    # ------------------------------------------------------------------

    def get_trust(self, source_type: InfoSourceType) -> float:
        trust_map = {
            InfoSourceType.GOVWEBSITE: self.trust_gov,
            InfoSourceType.X: self.trust_x,
            InfoSourceType.REDDIT: self.trust_reddit,
            InfoSourceType.FACEBOOK: self.trust_facebook,
        }
        return trust_map.get(source_type, 0.5)

    # ------------------------------------------------------------------
    # Agent Zero++ update methods
    # ------------------------------------------------------------------

    def update_probability(self, current_threat: float) -> None:
        """
        Update P as the rolling mean of recent threat observations.

        P = (1/m) * sum of last m threats
        """
        self.threat_memory.append(current_threat)
        if len(self.threat_memory) > self.memory_window:
            self.threat_memory.pop(0)
        self.probability = float(np.mean(self.threat_memory)) if self.threat_memory else 0.0

    def calculate_disposition(self, current_tick: int) -> float:
        """
        Compute D = (A + P + C + E*0.2) * amplifier - tau.

        The opinion-extremity amplifier (1 + beta*|opinion|) encodes
        motivational salience: extreme views heighten reactivity.
        """
        E = self.memory.get_avg_activation(current_tick) if self.memory else 0.0
        beta = 0.3
        amplifier = 1 + beta * abs(self.opinion)

        self.disposition = (
            (self.affect + self.probability + self.contagion + E * 0.2)
            * amplifier
            - self.tau
        )
        return self.disposition

    def update_state(self) -> None:
        """
        Determine behavioral state from disposition and affect.

        [MEDIUM] Flight/fight split uses age-modulated threshold (Berkowitz 1989;
                 Bracha 2004): impulse_control = 1 - (age-18)/100.
        [MEDIUM] PROTEST threshold is per-agent (Granovetter 1978): drawn from
                 N(0.3, 0.1) in _create_people.
        [HIGH]   RIOT->MOB threshold lowers when contagion is high
                 (Le Bon 1895; Lemos & Coelho 2015).
        """
        D, A = self.disposition, self.affect

        if D <= -0.2:
            self.state = AgentState.QUIET
        elif D <= 0.0:
            self.state = AgentState.AGITATED
        else:                          # D > 0: activated
            if A <= 0.5 * self.impulse_control:
                self.state = AgentState.FLIGHT
            elif D <= self.protest_threshold:
                self.state = AgentState.PROTEST
            else:
                riot_to_mob = self.riot_threshold - 0.3 * max(0.0, self.contagion - 0.5)
                if D <= riot_to_mob:
                    self.state = AgentState.RIOT
                else:
                    self.state = AgentState.MOB

    # ------------------------------------------------------------------
    # Visualisation helper
    # ------------------------------------------------------------------

    def get_color(self) -> str:
        if self.unhoused:
            return "#888780"
        return {
            AgentState.QUIET:    "#3B6D11",  # dark green
            AgentState.AGITATED: "#DAA520",  # goldenrod
            AgentState.FLIGHT:   "#185FA5",  # steel blue
            AgentState.PROTEST:  "#E69500",  # amber
            AgentState.RIOT:     "#D85A30",  # burnt orange
            AgentState.MOB:      "#A32D2D",  # dark red
        }.get(self.state, "#888780")
