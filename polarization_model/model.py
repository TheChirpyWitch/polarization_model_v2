"""
EnhancedPolarizationModel: the main ABM combining five mechanisms:

  1. Agent Zero++ — disposition-driven fight/flight behaviour
  2. ACT-R memory — power-law decay of information exposure
  3. Bounded-confidence opinion dynamics — convergence + divergence
  4. Schelling network segregation — homophilic rewiring -> echo chambers
  5. Info-source broadcast with trust — disinformation pathway
  6. Economic stress -> lower action threshold

Usage
-----
    from polarization_model.model import EnhancedPolarizationModel
    from polarization_model.parameters import ModelParameters

    model = EnhancedPolarizationModel(ModelParameters(num_people=100))
    model.setup()
    for _ in range(907):
        model.step()
"""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Callable, List, Optional

import numpy as np

from polarization_model.agents import ACTRMemory, AgentState, Person
from polarization_model.data_collector import DataCollector
from polarization_model.info_sources import InfoSource, InfoSourceType
from polarization_model.parameters import Location, ModelParameters


class EnhancedPolarizationModel:
    """
    Agent-based model of political polarisation under external media influence.

    The model runs on a discrete 2D grid.  Each tick corresponds to one
    observational unit of the EIM/NEIM time series (roughly one day at the
    default cadence used in the ICWSM experiments).

    Parameters
    ----------
    params : ModelParameters
        Simulation configuration.  Pass a customised instance to run
        experiments; defaults replicate the baseline reported in the paper.
    external_affect_fn : callable, optional
        f(tick) -> float in [-1, 1].  If provided, used to set the external
        affect signal instead of the default (constant zero).  Inject an
        EIM/NEIM signal here for empirically-driven runs.
    """

    def __init__(
        self,
        params: Optional[ModelParameters] = None,
        external_affect_fn: Optional[Callable[[int], float]] = None,
    ):
        self.params = params or ModelParameters()
        self.external_affect_fn = external_affect_fn or (lambda t: 0.0)

        self.tick: int = 0
        self.day_counter: int = 0
        self.cost_of_goods: float = 100.0

        self.people: List[Person] = []
        self.info_sources: List[InfoSource] = []
        self.work_locations: List[Location] = []
        self.school_locations: List[Location] = []
        self.home_locations: List[Location] = []

        self.min_coord = -32
        self.max_coord = 32

        self.data = DataCollector()
        self.is_setup = False

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def setup(self) -> None:
        """Initialise / reset the model and collect tick-0 data."""
        if self.params.seed is not None:
            random.seed(self.params.seed)
            np.random.seed(self.params.seed)

        self.tick = 0
        self.day_counter = 0
        self.cost_of_goods = 100.0
        self.people = []
        self.info_sources = []
        self.data.reset()

        self._setup_info_sources()
        self._setup_locations()
        self._create_people()
        self._create_social_network()

        self.data.collect(self)
        self.is_setup = True

    def step(self, skip_eim: bool = False) -> None:
        """Advance the simulation by one tick.

        Parameters
        ----------
        skip_eim : If True, the internal external_affect_fn is not applied
                   during this tick.  Set this when the caller has already
                   injected a source-specific signal to avoid double injection.
        """
        if not self.is_setup:
            return

        if self.tick % self.params.broadcast_interval == 0:
            self._broadcast_info()

        self._move_people()
        self._workplace_interactions()
        self._update_network()
        self._update_agent_zero(skip_eim=skip_eim)

        if self.tick % 2 == 0:
            self.day_counter += 1
            if self.day_counter % 30 == 0:
                self._monthly_econ()

        self.tick += 1
        self.data.collect(self)

    def run(self, steps: int) -> None:
        """Convenience wrapper: setup + run n steps."""
        self.setup()
        for _ in range(steps):
            self.step()

    # ------------------------------------------------------------------
    # Setup helpers
    # ------------------------------------------------------------------

    def _setup_info_sources(self) -> None:
        """
        Place four info sources with distinct stances and credibility.

        Source D is intentionally extreme (stance=0.8) to model a high-reach
        disinformation outlet.  Adjust configs here to test counterfactuals.
        """
        configs = [
            ("A", InfoSourceType.GOVWEBSITE, 0.0, 0.8),  # neutral, high credibility
            ("B", InfoSourceType.X, 0.6, 0.4),           # moderate pro-stance
            ("C", InfoSourceType.REDDIT, -0.5, 0.5),     # moderate anti-stance
            ("D", InfoSourceType.FACEBOOK, 0.8, 0.5),    # strong pro-stance (disinfo)
        ]
        y = self.max_coord - 1
        for sid, stype, stance, cred in configs:
            self.info_sources.append(InfoSource(
                source_id=sid,
                source_type=stype,
                x=random.randint(self.min_coord, self.max_coord),
                y=y,
                stance=stance,
                credibility=cred,
            ))

    def _setup_locations(self) -> None:
        """Create one workplace, one school, and N homes per quadrant."""
        quadrants = [
            (self.min_coord, -1, self.min_coord, -1),
            (0, self.max_coord, self.min_coord, -1),
            (self.min_coord, -1, 0, self.max_coord),
            (0, self.max_coord, 0, self.max_coord),
        ]
        used: set[tuple[int, int]] = set()

        for xmin, xmax, ymin, ymax in quadrants:
            loc = self._rand_loc(xmin, xmax, ymin, ymax, used)
            loc.location_type = "work"
            self.work_locations.append(loc)
            used.add((loc.x, loc.y))

        for xmin, xmax, ymin, ymax in quadrants:
            loc = self._rand_loc(xmin, xmax, ymin, ymax, used)
            loc.location_type = "school"
            self.school_locations.append(loc)
            used.add((loc.x, loc.y))

        for xmin, xmax, ymin, ymax in quadrants:
            for _ in range(self.params.homes_per_quadrant):
                loc = self._rand_loc(xmin, xmax, ymin, ymax, used)
                loc.location_type = "home"
                self.home_locations.append(loc)
                used.add((loc.x, loc.y))

    def _rand_loc(
        self, xmin: int, xmax: int, ymin: int, ymax: int, used: set
    ) -> Location:
        for _ in range(1000):
            x, y = random.randint(xmin, xmax), random.randint(ymin, ymax)
            if (x, y) not in used:
                return Location(x=x, y=y)
        return Location(x=random.randint(xmin, xmax), y=random.randint(ymin, ymax))

    def _create_people(self) -> None:
        p = self.params
        total = p.parts_govwebsite + p.parts_x + p.parts_reddit + p.parts_facebook
        p_gov = p.parts_govwebsite / total
        p_x = p.parts_x / total
        p_reddit = p.parts_reddit / total

        for i in range(p.num_people):
            agent = Person(unique_id=i)
            agent.memory = ACTRMemory(decay=p.memory_decay)

            agent.age = random.randint(18, 65)
            agent.savings = 1000.0
            agent.rent = 500.0

            agent.opinion = random.uniform(-1, 1)
            agent.gamma = random.uniform(0.1, 0.5)
            agent.threshold_pos = p.threshold_pos
            agent.threshold_neg = p.threshold_neg

            agent.trust_gov = float(np.clip(random.gauss(0.7, 0.15), 0.1, 1.0))
            agent.trust_x = float(np.clip(random.gauss(0.5, 0.2), 0.1, 1.0))
            agent.trust_reddit = float(np.clip(random.gauss(0.5, 0.2), 0.1, 1.0))
            agent.trust_facebook = float(np.clip(random.gauss(0.4, 0.2), 0.1, 1.0))

            # Younger agents have lower impulse control -> lower tau
            agent.impulse_control = 1.0 - (agent.age - 18) / 100
            agent.tau = p.tau_base + (agent.age - 18) * p.tau_age_factor

            # Per-agent activation thresholds (Granovetter 1978)
            agent.protest_threshold = float(np.clip(random.gauss(0.3, 0.1), 0.1, 0.5))
            agent.riot_threshold = float(np.clip(random.gauss(0.6, 0.1), 0.4, 0.9))

            agent.home_location = random.choice(self.home_locations)
            agent.x, agent.y = agent.home_location.x, agent.home_location.y

            if random.random() < 0.8:
                agent.employed = True
                agent.income = 2000.0
                agent.work_location = random.choice(self.work_locations)

            r = random.random()
            if r < p_gov:
                agent.info_preference = "govwebsite"
            elif r < p_gov + p_x:
                agent.info_preference = "X"
            elif r < p_gov + p_x + p_reddit:
                agent.info_preference = "reddit"
            else:
                agent.info_preference = "facebook"

            self.people.append(agent)

    def _create_social_network(self) -> None:
        """
        Wire initial connections with homophily bias.

        Half of each agent's connections are the most opinion-similar others;
        the remaining half are drawn randomly to avoid perfect echo chambers
        from the start.
        """
        p = self.params
        for agent in self.people:
            others = sorted(
                [o for o in self.people if o.unique_id != agent.unique_id],
                key=lambda o: abs(o.opinion - agent.opinion),
            )
            n_similar = p.initial_connections // 2
            n_random = p.initial_connections - n_similar

            connections: set[int] = set()
            for o in others[:n_similar]:
                connections.add(o.unique_id)

            random_pool = others[n_similar:]
            for o in random.sample(random_pool, min(n_random, len(random_pool))):
                connections.add(o.unique_id)

            agent.connections = list(connections)

    # ------------------------------------------------------------------
    # Per-tick mechanics
    # ------------------------------------------------------------------

    def _broadcast_info(self) -> None:
        """
        Push each info source's stance to agents.

        Agents who prefer the source are always exposed; all others have a
        20 % chance of algorithmic exposure.  Trust × credibility scales the
        effect.  Messages that oppose the agent's opinion trigger extra affect
        (outrage amplification).
        """
        for src in self.info_sources:
            msg_id, stance = src.broadcast(self.tick)

            for agent in self.people:
                pref_match = (agent.info_preference == src.source_type.value)
                if not pref_match and random.random() >= 0.2:
                    continue

                trust = agent.get_trust(src.source_type)
                agent.memory.add_exposure(msg_id, self.tick)

                alignment = 1 - abs(agent.opinion - stance)
                affect_delta = trust * src.credibility * 0.1
                if alignment < 0.3:
                    affect_delta *= 2.0   # outrage: opposing message
                elif alignment > 0.7:
                    affect_delta *= 1.5   # validation: confirming message
                affect_delta *= (1 + abs(stance))

                agent.affect = float(np.clip(agent.affect + affect_delta, 0, 1))

                opinion_shift = trust * src.credibility * (stance - agent.opinion) * 0.05
                agent.opinion = float(np.clip(agent.opinion + opinion_shift, -1, 1))

                threat = abs(stance) * trust * 0.5
                agent.update_probability(threat)

    def _move_people(self) -> None:
        for agent in self.people:
            if agent.unhoused or agent.state == AgentState.FLIGHT:
                angle = random.uniform(0, 2 * np.pi)
                agent.x = float(np.clip(agent.x + np.cos(angle) * 2, self.min_coord, self.max_coord))
                agent.y = float(np.clip(agent.y + np.sin(angle) * 2, self.min_coord, self.max_coord))
            elif agent.state in {AgentState.PROTEST, AgentState.RIOT, AgentState.MOB}:
                # Convergence toward centre simulates protest gathering.
                agent.x *= 0.95
                agent.y *= 0.95
            elif agent.employed:
                target = agent.work_location if self.tick % 2 == 0 else agent.home_location
                if target:
                    agent.x, agent.y = target.x, target.y

    def _workplace_interactions(self) -> None:
        """Bounded-confidence opinion update among co-located workers."""
        workers_at: dict[tuple[int, int], list[Person]] = defaultdict(list)
        for agent in self.people:
            if agent.employed and not agent.unhoused and agent.work_location:
                key = (int(agent.x), int(agent.y))
                if key == (agent.work_location.x, agent.work_location.y):
                    workers_at[key].append(agent)

        for workers in workers_at.values():
            if len(workers) < 2:
                continue
            for agent in workers:
                others = [w for w in workers if w.unique_id != agent.unique_id]
                if others:
                    self._interact(agent, random.choice(others))

    def _interact(self, p1: Person, p2: Person) -> None:
        """
        Apply bounded-confidence rule between two agents.

        Within threshold_pos -> converge; beyond threshold_neg -> diverge.
        The divergence formula preserves the [-1, 1] boundary while pushing
        agents toward their respective poles.
        """
        diff = abs(p1.opinion - p2.opinion)

        if diff <= p1.threshold_pos:
            x1, x2 = p1.opinion, p2.opinion
            p1.opinion = float(np.clip(x1 + p1.gamma * (x2 - x1), -1, 1))
            p2.opinion = float(np.clip(x2 + p2.gamma * (x1 - x2), -1, 1))

        if diff >= p1.threshold_neg:
            x1, x2 = p1.opinion, p2.opinion
            g1, g2 = p1.gamma, p2.gamma
            if x1 > x2:
                new1 = x1 + (g1 / 2) * (x1 - x2) * (1 - x1)
                new2 = x2 + (g2 / 2) * (x2 - x1) * (1 + x2)
            else:
                new1 = x1 + (g1 / 2) * (x1 - x2) * (1 + x1)
                new2 = x2 + (g2 / 2) * (x2 - x1) * (1 - x2)
            p1.opinion = float(np.clip(new1, -1, 1))
            p2.opinion = float(np.clip(new2, -1, 1))

    def _update_network(self) -> None:
        """
        Schelling-style network rewiring.

        Connections whose opinion gap exceeds segregation_tolerance are cut
        with probability rewire_prob and replaced with a similar-opinion agent.
        This drives homophily / echo-chamber formation over the run.
        """
        for agent in self.people:
            to_cut = [
                cid for cid in agent.connections
                if cid < len(self.people)
                and abs(agent.opinion - self.people[cid].opinion)
                > self.params.segregation_tolerance
            ]
            for cid in to_cut:
                if random.random() < self.params.rewire_prob:
                    agent.connections.remove(cid)
                    candidates = [
                        o for o in self.people
                        if o.unique_id != agent.unique_id
                        and o.unique_id not in agent.connections
                        and abs(o.opinion - agent.opinion) <= self.params.segregation_tolerance
                    ]
                    if candidates:
                        agent.connections.append(random.choice(candidates).unique_id)

            if agent.connections:
                similar = sum(
                    1 for cid in agent.connections
                    if cid < len(self.people)
                    and abs(self.people[cid].opinion - agent.opinion)
                    <= self.params.segregation_tolerance
                )
                agent.is_happy = (similar / len(agent.connections)) >= 0.5

    def _update_agent_zero(self, skip_eim: bool = False) -> None:
        """
        Update all Agent Zero++ components (C, A decay, tau, D, state).

        External affect is injected via external_affect_fn so the caller can
        swap between the weighted JSON signal and per-source NEIM signals
        without touching model internals.

        Parameters
        ----------
        skip_eim : If True, skip the external affect injection this tick.
                   Use when the caller has already applied a per-source signal.
        """
        alpha = 0.10
        external = self.external_affect_fn(self.tick)

        for agent in self.people:
            # Contagion: proportion of active network neighbours
            if agent.connections:
                active = sum(
                    1 for cid in agent.connections
                    if cid < len(self.people)
                    and self.people[cid].state in {
                        AgentState.AGITATED, AgentState.PROTEST,
                        AgentState.RIOT, AgentState.MOB,
                    }
                )
                agent.contagion = active / len(agent.connections)
            else:
                agent.contagion = 0.0

            # Affect decay blended with the external EIM signal
            if not skip_eim:
                agent.affect = (1 - alpha) * agent.affect * 0.95 + alpha * external
            else:
                agent.affect = agent.affect * 0.95

            # Economic hardship lowers the action threshold
            if agent.unhoused:
                econ_stress = self.params.tau_economic_factor
            elif agent.savings < 500:
                econ_stress = self.params.tau_economic_factor * 0.5
            else:
                econ_stress = 0.0

            agent.tau = max(
                0.2,
                self.params.tau_base
                + (agent.age - 18) * self.params.tau_age_factor
                - econ_stress,
            )

            agent.calculate_disposition(self.tick)
            agent.update_state()

    def _monthly_econ(self) -> None:
        """Deduct rent + goods, pay wages, check housing, apply inflation."""
        for agent in self.people:
            if not agent.unhoused:
                agent.savings = max(0.0, agent.savings - agent.rent - self.cost_of_goods)

        for agent in self.people:
            if agent.employed:
                agent.savings += agent.income

        for agent in self.people:
            if agent.savings <= 0:
                agent.unhoused = True
                agent.employed = False

        if self.params.increase_cost_of_goods:
            self.cost_of_goods *= self.params.cost_of_goods_growth_rate

    # ------------------------------------------------------------------
    # Reporting helpers
    # ------------------------------------------------------------------

    def print_status(self) -> None:
        print(f"\nTick: {self.tick}  |  Day: {self.day_counter}")

        print("\nEconomic:")
        print(f"  Cost of Goods: ${self.cost_of_goods:.2f}")
        print(f"  Employed: {self.data.count_employed[-1]}")
        print(f"  Unhoused: {self.data.count_unhoused[-1]}")

        print("\nAgent States:")
        print(f"  Quiet: {self.data.count_quiet[-1]}")
        print(f"  Agitated: {self.data.count_agitated[-1]}")
        print(f"  Flight: {self.data.count_flight[-1]}")
        print(f"  Protest: {self.data.count_protest[-1]}")
        print(f"  Riot: {self.data.count_riot[-1]}")
        print(f"  Mob: {self.data.count_mob[-1]}")

        print("\nOpinions:")
        print(f"  Mean: {self.data.opinion_mean[-1]:.4f}")
        print(f"  Std:  {self.data.opinion_std[-1]:.4f}")
        print(f"  Range: [{self.data.opinion_min[-1]:.4f}, {self.data.opinion_max[-1]:.4f}]")

        print("\nDisposition:")
        print(f"  Mean: {self.data.disposition_mean[-1]:.4f}")
        print(f"  Max:  {self.data.disposition_max[-1]:.4f}")

        print("\nNetwork:")
        print(f"  Avg connections: {self.data.avg_connections[-1]:.2f}")
        print(f"  Homophily: {self.data.network_homophily[-1]:.4f}")

    def print_agent_opinions(self, agent_ids: Optional[list[int]] = None) -> None:
        if agent_ids is None:
            agent_ids = list(range(min(10, len(self.people))))
        print(f"\nSample agents (tick {self.tick}):")
        for aid in agent_ids:
            if aid < len(self.people):
                p = self.people[aid]
                print(
                    f"  Agent {aid}: opinion={p.opinion:.3f}, "
                    f"affect={p.affect:.3f}, disp={p.disposition:.3f}, "
                    f"tau={p.tau:.3f}, state={p.state.value}"
                )
