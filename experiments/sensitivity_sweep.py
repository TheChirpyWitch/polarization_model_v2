"""
Parameter sensitivity sweep for the four core disposition-formula coefficients.

Sweeps memory_weight (lambda), opinion_amplifier (beta), affect_eim_alpha,
and affect_decay across a 3-point grid each, with N_SEEDS replications per
cell, for each of the three NEIM source signals (gov/news/reddit).

Output:
    sensitivity_results.csv  -- one row per (param, value, source, seed)
    sensitivity_summary.csv  -- aggregated mean +/- std per (param, value, source)

Run:
    python -m experiments.sensitivity_sweep \
        --gov    data/neim_1d_by_tick_gov_only.xlsx \
        --news   data/neim_1d_by_tick_news_only.xlsx \
        --reddit data/neim_1d_by_tick_reddit_only.xlsx \
        --output outputs/sensitivity \
        --n-seeds 5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from dataclasses import asdict
from typing import Callable

import numpy as np
import pandas as pd

# Allow running as `python experiments/sensitivity_sweep.py` from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from polarization_model.parameters import ModelParameters
from polarization_model.model import EnhancedPolarizationModel
from polarization_model.signals import load_excel_signal, get_signal
from polarization_model.agents import AgentState



# Configuration


SIM_START = 699   # 2021-01-02
SIM_END = 1606    # 2023-12-25
N_TICKS = SIM_END - SIM_START + 1

# Sensitivity grid: each parameter gets a 3-point grid centered on the default
SWEEP_GRID = {
    "memory_weight":     [0.1, 0.2, 0.4],   # lambda
    "opinion_amplifier": [0.0, 0.3, 0.6],   # beta (0.0 = no amplification)
    "affect_eim_alpha":  [0.05, 0.10, 0.20],
    "affect_decay":      [0.90, 0.95, 0.99],
}

DEFAULT_VALUES = {
    "memory_weight":     0.2,
    "opinion_amplifier": 0.3,
    "affect_eim_alpha":  0.10,
    "affect_decay":      0.95,
}

ACTIVE_STATES = {AgentState.PROTEST, AgentState.RIOT, AgentState.MOB}


# Single-cell run


def run_one(
    signal_fn: Callable[[int], float],
    overrides: dict,
    seed: int,
    num_people: int = 50,
) -> dict:
    """Run one full simulation and return summary metrics."""
    # Build parameters with default + overrides
    kwargs = {**DEFAULT_VALUES, **overrides, "num_people": num_people, "seed": seed}
    params = ModelParameters(**kwargs)

    model = EnhancedPolarizationModel(params=params, external_affect_fn=signal_fn)
    model.setup()
    model.tick = SIM_START

    disposition_traj = []
    fight_count = 0      # union of PROTEST/RIOT/MOB occupancy across ticks
    first_fight_tick = None

    for _ in range(N_TICKS):
        model.step()
        # Record metrics
        dispositions = [p.disposition for p in model.people]
        disposition_traj.append(np.mean(dispositions))

        active = sum(1 for p in model.people if p.state in ACTIVE_STATES)
        fight_count += active
        if active > 0 and first_fight_tick is None:
            first_fight_tick = model.tick

    return {
        "mean_disposition": float(np.mean(disposition_traj)),
        "max_disposition":  float(np.max(disposition_traj)),
        "fight_count":      fight_count,
        "first_fight_tick": first_fight_tick if first_fight_tick is not None else -1,
    }


# Sweep driver

def run_sweep(
    sources: dict,
    n_seeds: int,
    output_dir: Path,
    num_people: int = 50,
) -> None:
    """Sweep all parameters in SWEEP_GRID, all sources, all seeds."""
    rows = []

    total = sum(len(v) for v in SWEEP_GRID.values()) * len(sources) * n_seeds
    completed = 0
    print(f"Total runs to execute: {total}")

    for param_name, values in SWEEP_GRID.items():
        for value in values:
            overrides = {param_name: value}
            for source_name, signal_fn in sources.items():
                for seed in range(n_seeds):
                    metrics = run_one(
                        signal_fn=signal_fn,
                        overrides=overrides,
                        seed=seed,
                        num_people=num_people,
                    )
                    rows.append({
                        "param":  param_name,
                        "value":  value,
                        "source": source_name,
                        "seed":   seed,
                        **metrics,
                    })
                    completed += 1
                    if completed % 5 == 0 or completed == total:
                        print(f"  [{completed}/{total}] {param_name}={value} "
                              f"source={source_name} seed={seed}")

    # Save raw results
    df = pd.DataFrame(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / "sensitivity_results.csv"
    df.to_csv(raw_path, index=False)
    print(f"\nRaw results saved: {raw_path}")

    # Aggregate: mean +/- std across seeds, per (param, value, source)
    summary = (
        df.groupby(["param", "value", "source"])
          .agg(
              mean_disposition_avg=("mean_disposition", "mean"),
              mean_disposition_std=("mean_disposition", "std"),
              max_disposition_avg=("max_disposition", "mean"),
              max_disposition_std=("max_disposition", "std"),
              fight_count_avg=("fight_count", "mean"),
              fight_count_std=("fight_count", "std"),
              first_fight_tick_avg=("first_fight_tick", "mean"),
              first_fight_tick_std=("first_fight_tick", "std"),
          )
          .reset_index()
    )
    summary_path = output_dir / "sensitivity_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"Summary saved: {summary_path}")

    # Print human-readable summary
    print("\n" + "=" * 80)
    print("SENSITIVITY SWEEP SUMMARY")
    print("=" * 80)
    for param_name in SWEEP_GRID:
        sub = summary[summary["param"] == param_name]
        print(f"\nParameter: {param_name}")
        print(f"{'value':>8} {'source':>8} {'mean_D (avg+/-std)':>25} {'fight_count (avg)':>20}")
        for _, row in sub.iterrows():
            mean_d = f"{row['mean_disposition_avg']:.3f} +/- {row['mean_disposition_std']:.3f}"
            print(f"{row['value']:>8.2f} {row['source']:>8} {mean_d:>25} {row['fight_count_avg']:>20.0f}")


# CLI entrypoint

def main():
    parser = argparse.ArgumentParser(description="Parameter sensitivity sweep")
    parser.add_argument("--gov",    required=True, help="Gov NEIM xlsx path")
    parser.add_argument("--news",   required=True, help="News NEIM xlsx path")
    parser.add_argument("--reddit", required=True, help="Reddit NEIM xlsx path")
    parser.add_argument("--output", default="outputs/sensitivity", help="Output dir")
    parser.add_argument("--n-seeds", type=int, default=5, help="Replications per cell")
    parser.add_argument("--num-people", type=int, default=50, help="Population size")
    args = parser.parse_args()

    print("Loading NEIM signals...")
    gov_signal    = load_excel_signal(args.gov)
    news_signal   = load_excel_signal(args.news)
    reddit_signal = load_excel_signal(args.reddit)

    sources = {
        "gov":    lambda t: get_signal(gov_signal, t),
        "news":   lambda t: get_signal(news_signal, t),
        "reddit": lambda t: get_signal(reddit_signal, t),
    }

    run_sweep(
        sources=sources,
        n_seeds=args.n_seeds,
        output_dir=Path(args.output),
        num_people=args.num_people,
    )


if __name__ == "__main__":
    main()
