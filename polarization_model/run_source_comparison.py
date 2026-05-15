"""
Source comparison experiment.

Runs three parallel simulations — one each for Gov, News, and Reddit signal
conditions — and reports conflict onset metrics, Event A counts, and
generates comparison plots.

Usage
-----
Run from the repo root:
    python experiments/run_source_comparison.py

Or run cell-by-cell if pasted into a Jupyter notebook.
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from polarization_model import EnhancedPolarizationModel, ModelParameters
from polarization_model.agents import AgentState
from polarization_model.event_a import draw_event_a, DEFAULT_EVENT_A_PARAMS
from polarization_model.signals import load_excel_signal, get_signal

# ------------------------------------------------------------------
# 1. Configuration
# ------------------------------------------------------------------

# Path to the folder containing your data files
DATA_DIR = r'C:\Users\sherv\OneDrive\Desktop\CSSSA\GitHub Versions\polarization_model_v2\polarization_model_v2-main\data'

# Simulation window
# For Year 2021 - use the following: SIM_START = 699
#                                    SIM_END = 1162
# For Year 2022 - use the following: SIM_START = 1163
#                                    SIM_END = 1606

SIM_START  = 1163
SIM_END    = 1606
TOTAL_STEPS = SIM_END - SIM_START

# Affect injection weight
ALPHA = 0.10

# Plot colors per source
COLORS = {'gov': 'blue', 'news': 'orange', 'reddit': 'red'}

# ------------------------------------------------------------------
# 2. Load signals
# ------------------------------------------------------------------

signals = {
    'gov':    load_excel_signal(os.path.join(DATA_DIR, 'neim_1d_by_tick_gov_only.xlsx')),
    'news':   load_excel_signal(os.path.join(DATA_DIR, 'neim_1d_by_tick_news_only.xlsx')),
    'reddit': load_excel_signal(os.path.join(DATA_DIR, 'neim_1d_by_tick_reddit_only.xlsx')),
}

# ------------------------------------------------------------------
# 3. Results loop — scalar summary metrics
# ------------------------------------------------------------------

results = {}

for source_name in ['gov', 'news', 'reddit']:

    params = ModelParameters(
        num_people=50,
        tau_base=0.5,
        tau_age_factor=0.003,
        tau_economic_factor=0.25,
        segregation_tolerance=0.25,
        rewire_prob=0.15,
        broadcast_interval=5,
    )

    model = EnhancedPolarizationModel(params)
    model.setup()
    model.tick = SIM_START

    first_protest_tick = None
    first_riot_tick    = None
    first_mob_tick     = None
    protest_tick_count = 0
    riot_tick_count    = 0
    mob_tick_count     = 0

    total_protest_events  = 0
    total_riot_events     = 0
    total_mob_events      = 0

    total_event_a_protest = 0
    total_event_a_riot    = 0
    total_event_a_mob     = 0

    disp_means = []
    disp_maxes = []

    for i in range(TOTAL_STEPS):
        current_tick = model.tick
        ext = get_signal(signals[source_name], current_tick)

        # Inject per-source affect signal before stepping
        for p in model.people:
            p.affect = (1 - ALPHA) * p.affect * 0.95 + ALPHA * ext

        # skip_eim=True prevents double-injection inside _update_agent_zero
        model.step(skip_eim=True)

        disps = [p.disposition for p in model.people]
        disp_means.append(np.mean(disps))
        disp_maxes.append(np.max(disps))

        n_protest = sum(1 for p in model.people if p.state == AgentState.PROTEST)
        n_protest_with_intervention = sum(1 for p in model.people if p.state == AgentState.PROTEST_WITH_INTERVENTION)
        n_mob     = sum(1 for p in model.people if p.state == AgentState.MOB)

        if n_protest > 0:
            protest_tick_count += 1
            if first_protest_tick is None:
                first_protest_tick = current_tick

        if n_protest_with_intervention > 0:
            riot_tick_count += 1
            if first_riot_tick is None:
                first_riot_tick = current_tick

        if n_mob > 0:
            mob_tick_count += 1
            if first_mob_tick is None:
                first_mob_tick = current_tick

        total_protest_events += n_protest
        total_riot_events += n_protest_with_intervention
        total_mob_events     += n_mob

        # Independent Event A draw per agent per activated state
        tick_ea_protest = 0
        tick_ea_riot    = 0
        tick_ea_mob     = 0

        for p in model.people:
            if p.state == AgentState.PROTEST:
                tick_ea_protest += draw_event_a(AgentState.PROTEST,
                                                DEFAULT_EVENT_A_PARAMS[AgentState.PROTEST])
            elif p.state == AgentState.PROTEST_WITH_INTERVENTION:
                tick_ea_riot += draw_event_a(AgentState.PROTEST_WITH_INTERVENTION,
                                             DEFAULT_EVENT_A_PARAMS[AgentState.PROTEST_WITH_INTERVENTION])
            elif p.state == AgentState.MOB:
                tick_ea_mob += draw_event_a(AgentState.MOB,
                                            DEFAULT_EVENT_A_PARAMS[AgentState.MOB])

        total_event_a_protest += tick_ea_protest
        total_event_a_riot    += tick_ea_riot
        total_event_a_mob     += tick_ea_mob

    results[source_name] = {
        'mean_disp':              np.mean(disp_means),
        'max_disp':               np.max(disp_maxes),
        'first_protest':          first_protest_tick,
        'first_riot':             first_riot_tick,
        'first_mob':              first_mob_tick,
        'protests_per_month':     protest_tick_count   / (TOTAL_STEPS / 30),
        'riots_per_month':        riot_tick_count      / (TOTAL_STEPS / 30),
        'mobs_per_month':         mob_tick_count       / (TOTAL_STEPS / 30),
        'total_protest_events':   total_protest_events,
        'total_riot_events':      total_riot_events,
        'total_mob_events':       total_mob_events,
        'total_event_a_protest':  total_event_a_protest,
        'total_event_a_riot':     total_event_a_riot,
        'total_event_a_mob':      total_event_a_mob,
    }

# ------------------------------------------------------------------
# 4. Print comparison table
# ------------------------------------------------------------------

print(f"\n{'Metric':<30} {'Gov':>10} {'News':>10} {'Reddit':>10}")
print("-" * 60)

metrics = [
    ('Mean Disposition',       'mean_disp',             False),
    ('Max Disposition',        'max_disp',              False),
    ('First Protest Tick',     'first_protest',          True),
    ('First PWI Tick',      'first_riot',             True),
    ('First Mob Tick',         'first_mob',              True),
    ('Protests per Month',     'protests_per_month',    False),
    ('PWI per Month',       'riots_per_month',       False),
    ('Mobs per Month',         'mobs_per_month',        False),
    ('Total Protest Events',   'total_protest_events',  False),
    ('Total PWI Events',    'total_riot_events',     False),
    ('Total Mob Events',       'total_mob_events',      False),
]

for metric, key, is_tick in metrics:
    vals = [results[s][key] for s in ['gov', 'news', 'reddit']]
    if is_tick:
        print(f"{metric:<30} {str(vals[0]):>10} {str(vals[1]):>10} {str(vals[2]):>10}")
    else:
        print(f"{metric:<30} {vals[0]:>10.4f} {vals[1]:>10.4f} {vals[2]:>10.4f}")

# Conflict onset order
print("\nProtest onset order:")
order = sorted([(s, r['first_protest']) for s, r in results.items()
                if r['first_protest']], key=lambda x: x[1])
print("  " + " -> ".join(f"{s.upper()} (tick {t})" for s, t in order))

print("\nRiot onset order:")
order = sorted([(s, r['first_riot']) for s, r in results.items()
                if r['first_riot']], key=lambda x: x[1])
print("  " + " -> ".join(f"{s.upper()} (tick {t})" for s, t in order))

print("\nMob onset order:")
order = sorted([(s, r['first_mob']) for s, r in results.items()
                if r['first_mob']], key=lambda x: x[1])
print("  " + " -> ".join(f"{s.upper()} (tick {t})" for s, t in order))

# Agent-level state events summary
print(f"\n{'--- Total Agent-Level State Events by Source ---':^60}")
print(f"\n{'State':<15} {'Gov':>10} {'News':>10} {'Reddit':>10}")
print("-" * 45)
for state_label, key in [('Protest', 'total_protest_events'),
                          ('Riot',    'total_riot_events'),
                          ('Mob',     'total_mob_events')]:
    vals = [results[s][key] for s in ['gov', 'news', 'reddit']]
    print(f"{state_label:<15} {vals[0]:>10.0f} {vals[1]:>10.0f} {vals[2]:>10.0f}")

# Event A summary
print(f"\n{'--- Total Event A Count by State and Source ---':^60}")
print(f"\n{'State':<15} {'Gov':>10} {'News':>10} {'Reddit':>10}")
print("-" * 45)
for state_label, key in [('Protest', 'total_event_a_protest'),
                          ('Riot',    'total_event_a_riot'),
                          ('Mob',     'total_event_a_mob')]:
    vals = [results[s][key] for s in ['gov', 'news', 'reddit']]
    print(f"{state_label:<15} {vals[0]:>10.0f} {vals[1]:>10.0f} {vals[2]:>10.0f}")

# ------------------------------------------------------------------
# 5. Time-series loop — per-tick data for plotting
# ------------------------------------------------------------------

ts = {}
for src in ['gov', 'news', 'reddit']:
    model = EnhancedPolarizationModel(ModelParameters(
        num_people=50, tau_base=0.5, tau_age_factor=0.003,
        tau_economic_factor=0.25, segregation_tolerance=0.25,
        rewire_prob=0.15, broadcast_interval=5))
    model.setup()
    model.tick = SIM_START

    ts[src] = {
        'ticks':           [],
        'mean_d':          [],
        'max_d':           [],
        'mean_a':          [],
        'max_a':           [],
        'flight':          [],
        'protest':         [],
        'protest_with_intervention': [],
        'mob':             [],
        'event_a_protest': [],
        'event_a_protest_with_intervention':    [],
        'event_a_mob':     [],
        'draws_protest':   [],
        'draws_protest_with_intervention':      [],
        'draws_mob':       [],
    }

    for _ in range(TOTAL_STEPS):
        ext = get_signal(signals[src], model.tick)
        for p in model.people:
            p.affect = (1 - ALPHA) * p.affect * 0.95 + ALPHA * ext
        model.step(skip_eim=True)

        disps   = [p.disposition for p in model.people]
        affects = [p.affect      for p in model.people]

        ts[src]['ticks'].append(model.tick - 1)
        ts[src]['mean_d'].append(np.mean(disps))
        ts[src]['max_d'].append(np.max(disps))
        ts[src]['mean_a'].append(np.mean(affects))
        ts[src]['max_a'].append(np.max(affects))

        n_flight  = sum(1 for p in model.people if p.state == AgentState.FLIGHT)
        n_protest = sum(1 for p in model.people if p.state == AgentState.PROTEST)
        n_protest_with_intervention = sum(1 for p in model.people if p.state == AgentState.PROTEST_WITH_INTERVENTION)
        n_mob     = sum(1 for p in model.people if p.state == AgentState.MOB)

        ts[src]['flight'].append(n_flight)
        ts[src]['protest'].append(n_protest)
        ts[src]['protest_with_intervention'].append(n_protest_with_intervention)
        ts[src]['mob'].append(n_mob)

        tick_ea_protest = 0
        tick_ea_riot    = 0
        tick_ea_mob     = 0

        for p in model.people:
            if p.state == AgentState.PROTEST:
                draw = draw_event_a(AgentState.PROTEST,
                                    DEFAULT_EVENT_A_PARAMS[AgentState.PROTEST])
                ts[src]['draws_protest'].append(draw)
                tick_ea_protest += draw
            elif p.state == AgentState.PROTEST_WITH_INTERVENTION:
                draw = draw_event_a(AgentState.PROTEST_WITH_INTERVENTION,
                                    DEFAULT_EVENT_A_PARAMS[AgentState.PROTEST_WITH_INTERVENTION])
                ts[src]['draws_protest_with_intervention'].append(draw)
                tick_ea_riot += draw
            elif p.state == AgentState.MOB:
                draw = draw_event_a(AgentState.MOB,
                                    DEFAULT_EVENT_A_PARAMS[AgentState.MOB])
                ts[src]['draws_mob'].append(draw)
                tick_ea_mob += draw

        ts[src]['event_a_protest'].append(tick_ea_protest)
        ts[src]['event_a_protest_with_intervention'].append(tick_ea_riot)
        ts[src]['event_a_mob'].append(tick_ea_mob)

# ------------------------------------------------------------------
# 6. Figure 1 — 3x3 comparison plots
# ------------------------------------------------------------------

fig1, axes1 = plt.subplots(3, 3, figsize=(16, 12))

comparison_metrics = [
    ('mean_d',  'Mean Disposition'),
    ('max_d',   'Max Disposition'),
    ('mean_a',  'Mean Affect'),
    ('max_a',   'Max Affect'),
    ('flight',  'Flight Count'),
    ('protest', 'Protest Count'),
    ('protest_with_intervention', 'Protest w/ Intervention Count'),
    ('mob',     'Mob Count'),
]

for ax, (key, title) in zip(axes1.flat, comparison_metrics):
    for src in ['gov', 'news', 'reddit']:
        ax.plot(ts[src]['ticks'], ts[src][key], label=src.upper(),
                color=COLORS[src], alpha=0.8, linewidth=0.8)
    ax.set_xlabel('Tick')
    ax.set_title(title)
    ax.legend(fontsize=8)

axes1[2, 2].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(DATA_DIR, 'comparison_plot.png'), dpi=150)
plt.show()

# ------------------------------------------------------------------
# 7. Figure 2 — Event A frequency distributions + cumulative plots
# ------------------------------------------------------------------

fig2, axes2 = plt.subplots(2, 3, figsize=(16, 9))

state_keys = [
    ('draws_protest',  'event_a_protest', 'Protest', '#E69500'),
    ('draws_protest_with_intervention', 'event_a_protest_with_intervention', 'Protest w/ Intervention', '#D85A30'),
    ('draws_mob',      'event_a_mob',     'Mob',     '#A32D2D'),
]

# Row 1: frequency distributions
for col, (draw_key, _, state_label, color) in enumerate(state_keys):
    ax = axes2[0, col]
    all_draws = []
    for src in ['gov', 'news', 'reddit']:
        all_draws.extend(ts[src][draw_key])

    if all_draws:
        ax.hist(all_draws, bins=range(0, 102, 2),
                color=color, edgecolor='white', alpha=0.8, linewidth=0.4)
        ax.axvline(np.mean(all_draws), color='black', linestyle='--',
                   linewidth=1, label=f'Mean={np.mean(all_draws):.1f}')
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, 'No events recorded',
                ha='center', va='center', transform=ax.transAxes, fontsize=10)

    ax.set_xlabel('Event A count per draw')
    ax.set_ylabel('Frequency')
    ax.set_title(f'Event A Distribution — {state_label}')
    ax.set_xlim(0, 100)

# Row 2: cumulative Event A over time
for col, (_, cumul_key, state_label, color) in enumerate(state_keys):
    ax = axes2[1, col]
    for src in ['gov', 'news', 'reddit']:
        cumulative = np.cumsum(ts[src][cumul_key])
        ax.plot(ts[src]['ticks'], cumulative, label=src.upper(),
                color=COLORS[src], alpha=0.8, linewidth=0.9)
    ax.set_xlabel('Tick')
    ax.set_ylabel('Cumulative Event A count')
    ax.set_title(f'Cumulative Event A — {state_label}')
    ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig(os.path.join(DATA_DIR, 'event_a_plot.png'), dpi=150)
plt.show()

# ------------------------------------------------------------------
# 8. CSV export
# ------------------------------------------------------------------

rows = []
for src in ['gov', 'news', 'reddit']:
    for i in range(len(ts[src]['ticks'])):
        rows.append({
            'source':          src,
            'tick':            ts[src]['ticks'][i],
            'mean_disp':       ts[src]['mean_d'][i],
            'max_disp':        ts[src]['max_d'][i],
            'mean_aff':        ts[src]['mean_a'][i],
            'max_aff':         ts[src]['max_a'][i],
            'flight':          ts[src]['flight'][i],
            'protest':         ts[src]['protest'][i],
            'protest_with_intervention':            ts[src]['protest_with_intervention'][i],
            'mob':             ts[src]['mob'][i],
            'event_a_protest': ts[src]['event_a_protest'][i],
            'event_a_protest_with_intervention':    ts[src]['event_a_protest_with_intervention'][i],
            'event_a_mob':     ts[src]['event_a_mob'][i],
        })

pd.DataFrame(rows).to_csv(os.path.join(DATA_DIR, 'comparison_results.csv'), index=False)
print(f"\nCSV saved to: {os.path.join(DATA_DIR, 'comparison_results.csv')}")