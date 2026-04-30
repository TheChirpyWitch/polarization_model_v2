"""
Event A sampling utilities.

draw_event_a()  — draw how many discrete "Event A" occurrences follow from
                  a single agent being in a given behavioural state, using a
                  negative binomial distribution parameterised by (mu, r).

DEFAULT_EVENT_A_PARAMS — default mu/r values per AgentState, calibrated in v5:
    Protest  mu=3,  r=0.5  (low-intensity, high variance)
    Riot     mu=10, r=0.5
    Mob      mu=25, r=0.5  (high-intensity, heavy right tail)
"""

from typing import Optional

from scipy.stats import nbinom

from polarization_model.agents import AgentState


DEFAULT_EVENT_A_PARAMS: dict[AgentState, dict] = {
    AgentState.PROTEST: {"mu": 3,  "r": 0.5},
    AgentState.RIOT:    {"mu": 10, "r": 0.5},
    AgentState.MOB:     {"mu": 25, "r": 0.5},
}


def draw_event_a(state: AgentState, params: Optional[dict] = None) -> int:
    """
    Draw the number of Event A occurrences for one agent in *state*.

    Parameters
    ----------
    state  : The agent's current AgentState.  States not in params return 0.
    params : Dict with keys 'mu' (mean) and 'r' (dispersion).  Defaults to
             DEFAULT_EVENT_A_PARAMS[state] if omitted.

    Returns
    -------
    int in [0, 100]
    """
    if params is None:
        params = DEFAULT_EVENT_A_PARAMS.get(state)
    if params is None:
        return 0

    mu, r = params["mu"], params["r"]
    # scipy nbinom uses (n=r, p=r/(r+mu))
    p = r / (r + mu)
    return min(int(nbinom.rvs(n=r, p=p)), 100)
