"""
DataCollector: records per-tick population-level and per-agent metrics.

Call collect(model) once per tick.  After the run, get_summary_df() returns
a tidy DataFrame suitable for export or downstream analysis.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd

from polarization_model.agents import AgentState


class DataCollector:
    """Accumulates time-series data for a single simulation run."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.ticks: list[int] = []
        self.days: list[int] = []
        self.cost_of_goods: list[float] = []

        # Per-agent traces (keyed by unique_id)
        self.agent_opinions: dict[int, list[float]] = defaultdict(list)
        self.agent_dispositions: dict[int, list[float]] = defaultdict(list)
        self.agent_affects: dict[int, list[float]] = defaultdict(list)
        self.agent_states: dict[int, list[str]] = defaultdict(list)

        # Info-preference headcounts
        self.info_pref_counts: dict[str, list[int]] = defaultdict(list)

        # Economic headcounts
        self.count_employed: list[int] = []
        self.count_unhoused: list[int] = []

        # Behavioural-state headcounts
        self.count_quiet: list[int] = []
        self.count_agitated: list[int] = []
        self.count_fight: list[int] = []
        self.count_flight: list[int] = []

        # Opinion distribution statistics
        self.opinion_mean: list[float] = []
        self.opinion_std: list[float] = []
        self.opinion_min: list[float] = []
        self.opinion_max: list[float] = []

        # Disposition statistics
        self.disposition_mean: list[float] = []
        self.disposition_max: list[float] = []

        # Network statistics
        self.avg_connections: list[float] = []
        self.network_homophily: list[float] = []

    def collect(self, model) -> None:
        """Snapshot the current model state and append to all series."""
        self.ticks.append(model.tick)
        self.days.append(model.day_counter)
        self.cost_of_goods.append(model.cost_of_goods)

        opinions = []
        dispositions = []
        state_counts = {s: 0 for s in AgentState}
        pref_counts = {"govwebsite": 0, "X": 0, "reddit": 0, "facebook": 0}

        total_connections = 0
        homophily_sum = 0.0
        homophily_n = 0

        for p in model.people:
            self.agent_opinions[p.unique_id].append(p.opinion)
            self.agent_dispositions[p.unique_id].append(p.disposition)
            self.agent_affects[p.unique_id].append(p.affect)
            self.agent_states[p.unique_id].append(p.state.value)

            opinions.append(p.opinion)
            dispositions.append(p.disposition)
            state_counts[p.state] += 1
            pref_counts[p.info_preference] += 1

            total_connections += len(p.connections)
            for cid in p.connections:
                if cid < len(model.people):
                    similarity = 1 - abs(p.opinion - model.people[cid].opinion) / 2
                    homophily_sum += similarity
                    homophily_n += 1

        self.opinion_mean.append(float(np.mean(opinions)))
        self.opinion_std.append(float(np.std(opinions)))
        self.opinion_min.append(float(np.min(opinions)))
        self.opinion_max.append(float(np.max(opinions)))

        self.disposition_mean.append(float(np.mean(dispositions)))
        self.disposition_max.append(float(np.max(dispositions)))

        self.count_quiet.append(state_counts[AgentState.QUIET])
        self.count_agitated.append(state_counts[AgentState.AGITATED])
        self.count_fight.append(state_counts[AgentState.FIGHT])
        self.count_flight.append(state_counts[AgentState.FLIGHT])

        for pref, cnt in pref_counts.items():
            self.info_pref_counts[pref].append(cnt)

        self.count_employed.append(sum(1 for p in model.people if p.employed))
        self.count_unhoused.append(sum(1 for p in model.people if p.unhoused))

        n_people = len(model.people) or 1
        self.avg_connections.append(total_connections / n_people)
        self.network_homophily.append(homophily_sum / homophily_n if homophily_n > 0 else 0.0)

    def get_summary_df(self) -> pd.DataFrame:
        """Return population-level metrics as a tidy DataFrame."""
        return pd.DataFrame({
            "tick": self.ticks,
            "day": self.days,
            "cost_of_goods": self.cost_of_goods,
            "opinion_mean": self.opinion_mean,
            "opinion_std": self.opinion_std,
            "opinion_min": self.opinion_min,
            "opinion_max": self.opinion_max,
            "disposition_mean": self.disposition_mean,
            "disposition_max": self.disposition_max,
            "count_quiet": self.count_quiet,
            "count_agitated": self.count_agitated,
            "count_fight": self.count_fight,
            "count_flight": self.count_flight,
            "count_employed": self.count_employed,
            "count_unhoused": self.count_unhoused,
            "avg_connections": self.avg_connections,
            "homophily": self.network_homophily,
        })
