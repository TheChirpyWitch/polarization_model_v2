"""
Population size sensitivity sweep.

Tests whether the Government < News < Reddit ordering is robust to varying
the agent population size. Holds all four disposition coefficients at their
defaults (lambda=0.2, beta=0.3, alpha=0.10, decay=0.95) and varies only N.

Output:
    population_results.csv  -- one row per (N, source, seed)
    population_summary.csv  -- aggregated mean +/- std per (N, source)
    population_ordering.csv -- per-N check of gov<news<reddit ordering

Run:
    python -m experiments.population_sweep \
        --gov    data/neim_1d_by_tick_gov_only.xlsx \
        --news   data/neim_1d_by_tick_news_only.xlsx \
        --reddit data/neim_1d_by_tick_reddit_only.xlsx \
        --output outputs/population \
        --n-seeds 10

Default population grid: [50, 100, 200, 500]
This covers a 10x range in population size. The reported simulations
in the main paper use N=50.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from polarization_model.parameters import ModelParameters
from polarization_model.model import EnhancedPolarizationModel
from polarization_model.signals import load_excel_signal, get_signal
from polarization_model.agents import AgentState


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SIM_START = 699   # 2021-01-02
SIM_END = 1606    # 2023-12-25
N_TICKS = SIM_END - SIM_START + 1

# Population sizes to sweep
POPULATION_GRID = [50, 100, 200, 500]

# Default disposition coefficients (same as paper's main results)
DEFAULT_PARAMS = {
    "memory_weight":     0.2,
    "opinion_amplifier": 0.3,
    "affect_eim_alpha":  0.10,
    "affect_decay":      0.95,
}

ACTIVE_STATES = {AgentState.PROTEST, AgentState.RIOT, AgentState.MOB}


# ---------------------------------------------------------------------------
# Single run
# ---------------------------------------------------------------------------

def run_one(
    signal_fn: Callable[[int], float],
    num_people: int,
    seed: int,
) -> dict:
    """Run one full simulation and return summary metrics."""
    kwargs = {**DEFAULT_PARAMS, "num_people": num_people, "seed": seed}
    params = ModelParameters(**kwargs)

    model = EnhancedPolarizationModel(params=params, external_affect_fn=signal_fn)
    model.setup()
    model.tick = SIM_START

    disposition_traj = []
    fight_count = 0
    first_fight_tick = None

    for _ in range(N_TICKS):
        model.step()
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


# ---------------------------------------------------------------------------
# Sweep driver
# ---------------------------------------------------------------------------

def run_population_sweep(
    sources: dict,
    n_seeds: int,
    output_dir: Path,
) -> None:
    """Sweep population sizes across all sources and seeds."""
    rows = []

    total = len(POPULATION_GRID) * len(sources) * n_seeds
    completed = 0
    print(f"Population sweep: {len(POPULATION_GRID)} N values x {len(sources)} sources x {n_seeds} seeds = {total} runs")
    print(f"  Population grid: {POPULATION_GRID}")
    print(f"  Defaults: {DEFAULT_PARAMS}")

    for num_people in POPULATION_GRID:
        print(f"\n  >>> Starting N={num_people} <<<")
        for source_name, signal_fn in sources.items():
            for seed in range(n_seeds):
                metrics = run_one(
                    signal_fn=signal_fn,
                    num_people=num_people,
                    seed=seed,
                )
                rows.append({
                    "num_people": num_people,
                    "source":     source_name,
                    "seed":       seed,
                    **metrics,
                })
                completed += 1
                if completed % 5 == 0 or completed == total:
                    print(f"  [{completed}/{total}] N={num_people} "
                          f"source={source_name} seed={seed}")

    df = pd.DataFrame(rows)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_path = output_dir / "population_results.csv"
    df.to_csv(raw_path, index=False)
    print(f"\nRaw results saved: {raw_path}")

    # Aggregate per (N, source)
    summary = (
        df.groupby(["num_people", "source"])
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
    summary_path = output_dir / "population_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"Summary saved: {summary_path}")

    # Per-N ordering check
    ordering_rows = []
    for n in POPULATION_GRID:
        cell_summary = summary[summary["num_people"] == n]
        if len(cell_summary) != 3:
            continue
        gov = cell_summary[cell_summary["source"] == "gov"]["mean_disposition_avg"].iloc[0]
        news = cell_summary[cell_summary["source"] == "news"]["mean_disposition_avg"].iloc[0]
        reddit = cell_summary[cell_summary["source"] == "reddit"]["mean_disposition_avg"].iloc[0]

        gov_std = cell_summary[cell_summary["source"] == "gov"]["mean_disposition_std"].iloc[0]
        reddit_std = cell_summary[cell_summary["source"] == "reddit"]["mean_disposition_std"].iloc[0]
        pooled_std = float(np.sqrt(gov_std**2 + reddit_std**2))
        cohen_d = (reddit - gov) / pooled_std if pooled_std > 0 else float('inf')

        ordering_rows.append({
            "num_people":      n,
            "gov":             gov,
            "news":            news,
            "reddit":          reddit,
            "ordering_holds":  (gov < news < reddit),
            "reddit_minus_gov": reddit - gov,
            "cohen_d":         cohen_d,
        })

    ordering_df = pd.DataFrame(ordering_rows)
    ordering_path = output_dir / "population_ordering.csv"
    ordering_df.to_csv(ordering_path, index=False)
    print(f"Ordering check saved: {ordering_path}")

    # Print conclusions
    n_pass = ordering_df["ordering_holds"].sum()
    print()
    print("=" * 80)
    print("POPULATION SIZE SENSITIVITY CONCLUSIONS")
    print("=" * 80)
    print(f"  Population values tested: {POPULATION_GRID}")
    print(f"  Population values where gov<news<reddit holds: {n_pass}/{len(POPULATION_GRID)}")
    print()
    print("  Per-N summary:")
    print(f"  {'N':>5} {'gov':>10} {'news':>10} {'reddit':>10} {'r-g':>8} {'Cohen d':>10}")
    for _, row in ordering_df.iterrows():
        mark = "✓" if row['ordering_holds'] else "✗"
        print(f"  {int(row['num_people']):>5} "
              f"{row['gov']:>+10.3f} {row['news']:>+10.3f} {row['reddit']:>+10.3f} "
              f"{row['reddit_minus_gov']:>+8.3f} {row['cohen_d']:>10.1f}  [{mark}]")


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Population size sensitivity sweep")
    parser.add_argument("--gov",    required=True, help="Gov NEIM xlsx path")
    parser.add_argument("--news",   required=True, help="News NEIM xlsx path")
    parser.add_argument("--reddit", required=True, help="Reddit NEIM xlsx path")
    parser.add_argument("--output", default="outputs/population", help="Output dir")
    parser.add_argument("--n-seeds", type=int, default=10, help="Replications per cell")
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

    run_population_sweep(
        sources=sources,
        n_seeds=args.n_seeds,
        output_dir=Path(args.output),
    )


if __name__ == "__main__":
    main()
