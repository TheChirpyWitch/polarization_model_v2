"""
Full-factorial sensitivity sweep over the four core disposition coefficients.

Sweeps memory_weight (lambda), opinion_amplifier (beta), affect_eim_alpha,
and affect_decay simultaneously across a 3-point grid each (3^4 = 81 cells),
with N_SEEDS replications per cell, for each of the three NEIM source
signals (gov/news/reddit).

This complements the OAT (one-at-a-time) sweep in sensitivity_sweep.py.
Full-factorial reveals interaction effects that OAT cannot detect, which
is important for models with multiplicative structure (e.g., the
disposition formula's (...)*(1+beta*|opinion|) amplifier).

Output:
    factorial_results.csv  -- one row per (cell, source, seed)
    factorial_summary.csv  -- aggregated mean +/- std per (cell, source)
    factorial_ordering.csv -- per-cell check of gov<news<reddit ordering

Run:
    python -m experiments.factorial_sweep \
        --gov    data/neim_1d_by_tick_gov_only.xlsx \
        --news   data/neim_1d_by_tick_news_only.xlsx \
        --reddit data/neim_1d_by_tick_reddit_only.xlsx \
        --output outputs/factorial \
        --n-seeds 10
"""

from __future__ import annotations

import argparse
import sys
import itertools
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

# Allow running as `python experiments/factorial_sweep.py` from project root
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

# Same 3-point grid as the OAT sweep, but now sweep ALL combinations
PARAM_GRID = {
    "memory_weight":     [0.1, 0.2, 0.4],
    "opinion_amplifier": [0.0, 0.3, 0.6],
    "affect_eim_alpha":  [0.05, 0.10, 0.20],
    "affect_decay":      [0.90, 0.95, 0.99],
}

PARAM_NAMES = list(PARAM_GRID.keys())  # fixed order
ACTIVE_STATES = {AgentState.PROTEST, AgentState.RIOT, AgentState.MOB}


# ---------------------------------------------------------------------------
# Single run
# ---------------------------------------------------------------------------

def run_one(
    signal_fn: Callable[[int], float],
    param_values: dict,
    seed: int,
    num_people: int = 50,
) -> dict:
    """Run one full simulation and return summary metrics."""
    kwargs = {**param_values, "num_people": num_people, "seed": seed}
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

def run_factorial(
    sources: dict,
    n_seeds: int,
    output_dir: Path,
    num_people: int = 50,
) -> None:
    """Sweep all combinations of all parameters, all sources, all seeds."""
    rows = []

    # Generate all parameter combinations (3^4 = 81 cells)
    grids = [PARAM_GRID[name] for name in PARAM_NAMES]
    all_combinations = list(itertools.product(*grids))
    n_cells = len(all_combinations)

    total = n_cells * len(sources) * n_seeds
    completed = 0
    print(f"Full-factorial sweep: {n_cells} parameter cells x {len(sources)} sources x {n_seeds} seeds = {total} runs")

    for cell_idx, combo in enumerate(all_combinations):
        param_values = dict(zip(PARAM_NAMES, combo))

        for source_name, signal_fn in sources.items():
            for seed in range(n_seeds):
                metrics = run_one(
                    signal_fn=signal_fn,
                    param_values=param_values,
                    seed=seed,
                    num_people=num_people,
                )
                rows.append({
                    "cell_idx": cell_idx,
                    **param_values,
                    "source":   source_name,
                    "seed":     seed,
                    **metrics,
                })
                completed += 1
                if completed % 30 == 0 or completed == total:
                    print(f"  [{completed}/{total}] cell={cell_idx+1}/{n_cells} "
                          f"source={source_name}")

    df = pd.DataFrame(rows)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_path = output_dir / "factorial_results.csv"
    df.to_csv(raw_path, index=False)
    print(f"\nRaw results saved: {raw_path}")

    # Aggregate per (cell, source)
    summary = (
        df.groupby(["cell_idx"] + PARAM_NAMES + ["source"])
          .agg(
              mean_disposition_avg=("mean_disposition", "mean"),
              mean_disposition_std=("mean_disposition", "std"),
              max_disposition_avg=("max_disposition", "mean"),
              max_disposition_std=("max_disposition", "std"),
              fight_count_avg=("fight_count", "mean"),
              fight_count_std=("fight_count", "std"),
          )
          .reset_index()
    )
    summary_path = output_dir / "factorial_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"Summary saved: {summary_path}")

    # Per-cell ordering check
    ordering_rows = []
    for cell_idx in range(n_cells):
        cell_summary = summary[summary["cell_idx"] == cell_idx]
        if len(cell_summary) != 3:  # expect 3 sources
            continue
        gov = cell_summary[cell_summary["source"] == "gov"]["mean_disposition_avg"].iloc[0]
        news = cell_summary[cell_summary["source"] == "news"]["mean_disposition_avg"].iloc[0]
        reddit = cell_summary[cell_summary["source"] == "reddit"]["mean_disposition_avg"].iloc[0]

        gov_std = cell_summary[cell_summary["source"] == "gov"]["mean_disposition_std"].iloc[0]
        reddit_std = cell_summary[cell_summary["source"] == "reddit"]["mean_disposition_std"].iloc[0]
        pooled_std = float(np.sqrt(gov_std**2 + reddit_std**2))
        cohen_d = (reddit - gov) / pooled_std if pooled_std > 0 else float('inf')

        param_dict = {name: cell_summary.iloc[0][name] for name in PARAM_NAMES}
        ordering_rows.append({
            "cell_idx": cell_idx,
            **param_dict,
            "gov":           gov,
            "news":          news,
            "reddit":        reddit,
            "ordering_holds": (gov < news < reddit),
            "reddit_minus_gov": reddit - gov,
            "cohen_d":       cohen_d,
        })

    ordering_df = pd.DataFrame(ordering_rows)
    ordering_path = output_dir / "factorial_ordering.csv"
    ordering_df.to_csv(ordering_path, index=False)
    print(f"Ordering check saved: {ordering_path}")

    # Print top-line conclusions
    n_pass = ordering_df["ordering_holds"].sum()
    print()
    print("=" * 80)
    print("FULL-FACTORIAL SWEEP CONCLUSIONS")
    print("=" * 80)
    print(f"  Total cells:                            {n_cells}")
    print(f"  Cells where gov<news<reddit holds:      {n_pass}/{n_cells}")
    print(f"  Min (reddit - gov) margin across cells: {ordering_df['reddit_minus_gov'].min():+.3f}")
    print(f"  Max (reddit - gov) margin across cells: {ordering_df['reddit_minus_gov'].max():+.3f}")
    print(f"  Min Cohen's d across cells:             {ordering_df['cohen_d'].min():.1f}")
    print(f"  Max Cohen's d across cells:             {ordering_df['cohen_d'].max():.1f}")
    if n_pass < n_cells:
        broken = ordering_df[~ordering_df["ordering_holds"]]
        print(f"\n  Cells where ordering breaks ({len(broken)}):")
        for _, row in broken.iterrows():
            print(f"    cell {int(row['cell_idx']):3d}: "
                  f"mw={row['memory_weight']}, oa={row['opinion_amplifier']}, "
                  f"a={row['affect_eim_alpha']}, decay={row['affect_decay']}: "
                  f"gov={row['gov']:+.3f}, news={row['news']:+.3f}, reddit={row['reddit']:+.3f}")


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Full-factorial sensitivity sweep")
    parser.add_argument("--gov",    required=True, help="Gov NEIM xlsx path")
    parser.add_argument("--news",   required=True, help="News NEIM xlsx path")
    parser.add_argument("--reddit", required=True, help="Reddit NEIM xlsx path")
    parser.add_argument("--output", default="outputs/factorial", help="Output dir")
    parser.add_argument("--n-seeds", type=int, default=10, help="Replications per cell")
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

    run_factorial(
        sources=sources,
        n_seeds=args.n_seeds,
        output_dir=Path(args.output),
        num_people=args.num_people,
    )


if __name__ == "__main__":
    main()
