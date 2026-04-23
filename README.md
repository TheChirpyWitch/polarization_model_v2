# ICWSM Polarization Model

Agent-based model of political polarisation under empirical media influence, combining:

- **Agent Zero++** — disposition-driven fight / flight behaviour (Affect + Probability + Contagion > threshold)
- **ACT-R memory** — power-law decay of information exposure history
- **Bounded-confidence opinion dynamics** — convergence within tolerance, divergence beyond it
- **Schelling network segregation** — homophilic rewiring that produces echo chambers over time
- **EIM / NEIM signal injection** — external affect driven by empirical media signals (weighted PCA or per-source)
- **Economic stress** — inflation and income shocks that lower the action threshold

---

## Directory layout

```
.
├── polarization_model/        # Core package (import from here)
│   ├── __init__.py
│   ├── agents.py              # AgentState, ACTRMemory, Person
│   ├── data_collector.py      # DataCollector
│   ├── info_sources.py        # InfoSourceType, InfoSource
│   ├── model.py               # EnhancedPolarizationModel
│   ├── parameters.py          # ModelParameters, Location
│   ├── signals.py             # Signal loaders and accessor
│   └── visualization.py       # plot_results, export_data
├── experiments/
│   ├── run_baseline.py        # Single run with weighted EIM JSON
│   └── run_comparison.py      # Gov / news / reddit NEIM comparison
├── data/                      # Place input data files here (not tracked)
├── outputs/                   # Artefacts written here (not tracked)
├── requirements.txt
└── README.md
```

---

## Setup

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .                   # installs polarization_model as editable package
```

> Python 3.10+ is required (uses `str | Path` union type hints).

---

## Data files

Place the following files in the `data/` directory before running experiments:

| File | Description | Used by |
|------|-------------|---------|
| `eim_1d_by_tick_v8_weighted_general_topic_general_narrative.json` | Weighted PCA EIM signal (ticks 1–1606) | `run_baseline.py` |
| `neim_1d_by_tick_gov_only.xlsx` | Gov-only NEIM signal (columns: Tick, NEIM_1D_Value) | `run_comparison.py` |
| `neim_1d_by_tick_news_only.xlsx` | News-only NEIM signal | `run_comparison.py` |
| `neim_1d_by_tick_reddit_only.xlsx` | Reddit-only NEIM signal | `run_comparison.py` |

---

## Running experiments

### Baseline (weighted signal)

```bash
python -m experiments.run_baseline \
    --signal data/eim_1d_by_tick_v8_weighted_general_topic_general_narrative.json \
    --output outputs/baseline \
    --num-people 50 \
    --seed 42
```

Outputs written to `outputs/`:
- `baseline_summary.csv` — tick-level population metrics
- `baseline_results.png` — 3×3 diagnostic panel

### Multi-source comparison

```bash
python -m experiments.run_comparison \
    --gov    data/neim_1d_by_tick_gov_only.xlsx \
    --news   data/neim_1d_by_tick_news_only.xlsx \
    --reddit data/neim_1d_by_tick_reddit_only.xlsx \
    --output outputs/comparison \
    --num-people 50 \
    --seed 42
```

Outputs written to `outputs/`:
- `comparison_comparison_results.csv` — long-format per-tick data for all three sources
- `comparison_comparison_plot.png` — 2×2 panel comparing disposition and conflict counts

---

## Simulation period

Both experiments default to ticks **699 → 1606** (2021-01-02 to 2023-12-25), the period covered by the v8 EIM/NEIM signals.  Change `SIM_START` / `SIM_END` at the top of each experiment script to extend or narrow the window.

---

## Key parameters

All knobs live in `ModelParameters` (`polarization_model/parameters.py`).  Pass a customised instance to `EnhancedPolarizationModel` to run counterfactuals without touching model internals.

| Parameter | Default | Effect |
|-----------|---------|--------|
| `num_people` | 50 | Population size |
| `tau_base` | 0.6 | Base action threshold (lower → easier to trigger conflict) |
| `tau_age_factor` | 0.005 | Older agents have higher threshold |
| `tau_economic_factor` | 0.2 | Economic hardship lowers threshold |
| `threshold_pos` | 0.4 | Opinion gap below which agents converge |
| `threshold_neg` | 0.6 | Opinion gap above which agents diverge (polarise) |
| `segregation_tolerance` | 0.3 | Max opinion diff tolerated in social network |
| `rewire_prob` | 0.1 | Probability of cutting a too-dissimilar connection |
| `broadcast_interval` | 10 | Ticks between info-source broadcasts |
| `memory_decay` | 0.5 | ACT-R decay exponent (0.5 = standard literature value) |
| `cost_of_goods_growth_rate` | 1.8 | Monthly inflation multiplier |

### Example: low-threshold sensitivity run

```python
from polarization_model import EnhancedPolarizationModel, ModelParameters
from polarization_model.signals import load_json_signal, get_signal

signal = load_json_signal("data/eim_1d_by_tick_v8_weighted_general_topic_general_narrative.json")

params = ModelParameters(
    num_people=100,
    tau_base=0.3,               # lower threshold
    segregation_tolerance=0.2,  # stricter echo chambers
)

model = EnhancedPolarizationModel(
    params=params,
    external_affect_fn=lambda tick: get_signal(signal, tick),
)
model.setup()
model.tick = 699
for _ in range(907):
    model.step()

model.print_status()
```

---

## Extending the model

### Adding a new info source

In `model.py → _setup_info_sources()`, append a tuple to `configs`:

```python
("E", InfoSourceType.REDDIT, -0.9, 0.6),  # extreme anti-stance source
```

### Injecting a custom signal

Pass any callable `f(tick: int) -> float` as `external_affect_fn`:

```python
import math
model = EnhancedPolarizationModel(
    external_affect_fn=lambda t: math.sin(t / 50) * 0.5
)
```

### Collecting additional per-agent metrics

Extend `DataCollector.collect()` in `data_collector.py` and add matching columns to `get_summary_df()`.

---

## Reproducing ICWSM paper results

1. Copy data files to `data/` as listed above.
2. Run `run_comparison.py` with `--seed 42` and `--num-people 50`.
3. The console table and `comparison_comparison_results.csv` contain the disposition and conflict-onset metrics reported in the paper.

Expected console output (approximate, stochastic):

```
Metric                           Gov       News     Reddit
-------------------------------------------------------
Mean Disposition              0.0986     0.1555     0.3718
Max Disposition               1.3739     1.8082     1.9538
First Fight Tick                None       1430        875
Fights per Month              0.0000     2.9438    11.9735

Conflict order: REDDIT (tick 875) -> NEWS (tick 1430)
```
