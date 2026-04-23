"""
Multi-source comparison experiment.

Runs three independent simulations — one each for gov-only, news-only, and
reddit-only NEIM signals — and reports disposition and conflict metrics side
by side.  Also saves a 2×2 comparison plot and a long-format CSV.

Usage
-----
    python -m experiments.run_comparison \
        --gov   data/neim_1d_by_tick_gov_only.xlsx \
        --news  data/neim_1d_by_tick_news_only.xlsx \
        --reddit data/neim_1d_by_tick_reddit_only.xlsx \
        --output outputs/comparison
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from polarization_model.agents import AgentState
from polarization_model.model import EnhancedPolarizationModel
from polarization_model.parameters import ModelParameters
from polarization_model.signals import get_signal, load_excel_signal

SIM_START = 699
SIM_END = 1606
ALPHA = 0.10

COLORS = {"gov": "blue", "news": "orange", "reddit": "red"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare gov / news / reddit NEIM signals.")
    parser.add_argument("--gov",    default="data/neim_1d_by_tick_gov_only.xlsx")
    parser.add_argument("--news",   default="data/neim_1d_by_tick_news_only.xlsx")
    parser.add_argument("--reddit", default="data/neim_1d_by_tick_reddit_only.xlsx")
    parser.add_argument("--output", default="outputs/comparison")
    parser.add_argument("--num-people", type=int, default=50)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def build_params(num_people: int) -> ModelParameters:
    return ModelParameters(
        num_people=num_people,
        tau_base=0.5,
        tau_age_factor=0.003,
        tau_economic_factor=0.25,
        segregation_tolerance=0.25,
        rewire_prob=0.15,
        broadcast_interval=5,
    )


def run_one(source_name: str, signal: dict, params: ModelParameters) -> tuple[dict, dict]:
    """
    Run a single simulation with the given NEIM signal.

    Returns
    -------
    summary : scalar metrics dict
    timeseries : per-tick series dict
    """
    model = EnhancedPolarizationModel(
        params=params,
        external_affect_fn=lambda tick: get_signal(signal, tick),
    )
    model.setup()
    model.tick = SIM_START

    total_steps = SIM_END - SIM_START
    ts: dict[str, list] = {"ticks": [], "mean_d": [], "max_d": [], "fight": [], "flight": []}
    first_fight_tick = None
    fight_tick_count = 0

    for _ in range(total_steps):
        model.step()
        t = model.tick - 1

        disps = [p.disposition for p in model.people]
        n_fight = sum(1 for p in model.people if p.state == AgentState.FIGHT)

        ts["ticks"].append(t)
        ts["mean_d"].append(float(np.mean(disps)))
        ts["max_d"].append(float(np.max(disps)))
        ts["fight"].append(n_fight)
        ts["flight"].append(sum(1 for p in model.people if p.state == AgentState.FLIGHT))

        if n_fight > 0:
            fight_tick_count += 1
            if first_fight_tick is None:
                first_fight_tick = t

    summary = {
        "mean_disp": float(np.mean(ts["mean_d"])),
        "max_disp": float(np.max(ts["max_d"])),
        "first_fight": first_fight_tick,
        "fights_per_month": fight_tick_count / (total_steps / 30),
    }
    return summary, ts


def print_comparison_table(results: dict) -> None:
    sources = ["gov", "news", "reddit"]
    header = f"{'Metric':<25} {'Gov':>10} {'News':>10} {'Reddit':>10}"
    print(f"\n{header}")
    print("-" * 55)
    for label, key in [
        ("Mean Disposition",  "mean_disp"),
        ("Max Disposition",   "max_disp"),
        ("First Fight Tick",  "first_fight"),
        ("Fights per Month",  "fights_per_month"),
    ]:
        vals = [results[s][key] for s in sources]
        if key == "first_fight":
            row = [str(v) if v is not None else "None" for v in vals]
            print(f"{label:<25} {row[0]:>10} {row[1]:>10} {row[2]:>10}")
        else:
            print(f"{label:<25} {vals[0]:>10.4f} {vals[1]:>10.4f} {vals[2]:>10.4f}")

    ordered = sorted(
        [(s, r["first_fight"]) for s, r in results.items() if r["first_fight"] is not None],
        key=lambda x: x[1],
    )
    if ordered:
        print(f"\nConflict order: {' -> '.join(f'{s.upper()} (tick {t})' for s, t in ordered)}")
    else:
        print("\nNo FIGHT events observed in any condition.")


def save_comparison_plot(timeseries: dict, output_prefix: str) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    metrics = [
        ("mean_d", "Mean Disposition"),
        ("max_d",  "Max Disposition"),
        ("fight",  "Fight Count"),
        ("flight", "Flight Count"),
    ]
    for ax, (key, title) in zip(axes.flat, metrics):
        for src, ts in timeseries.items():
            ax.plot(ts["ticks"], ts[key], label=src.upper(),
                    color=COLORS[src], alpha=0.8, linewidth=0.8)
        ax.set_xlabel("Tick")
        ax.set_title(title)
        ax.legend()
    plt.tight_layout()
    path = f"{output_prefix}_comparison_plot.png"
    plt.savefig(path, dpi=150)
    print(f"Saved figure: {path}")


def save_comparison_csv(timeseries: dict, output_prefix: str) -> None:
    rows = []
    for src, ts in timeseries.items():
        for i in range(len(ts["ticks"])):
            rows.append({
                "source": src,
                "tick": ts["ticks"][i],
                "mean_disp": ts["mean_d"][i],
                "max_disp": ts["max_d"][i],
                "fight": ts["fight"][i],
                "flight": ts["flight"][i],
            })
    path = f"{output_prefix}_comparison_results.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"Saved data: {path}")


def main() -> None:
    args = parse_args()

    if args.seed is not None:
        import random
        random.seed(args.seed)
        np.random.seed(args.seed)

    signal_files = {"gov": args.gov, "news": args.news, "reddit": args.reddit}
    print("Loading NEIM signals...")
    signals = {name: load_excel_signal(path) for name, path in signal_files.items()}

    params = build_params(args.num_people)
    results: dict = {}
    timeseries: dict = {}

    for source_name, signal in signals.items():
        print(f"\n[{source_name.upper()}] running {SIM_END - SIM_START} ticks...")
        summary, ts = run_one(source_name, signal, params)
        results[source_name] = summary
        timeseries[source_name] = ts

    print_comparison_table(results)

    output_prefix = args.output
    Path(output_prefix).parent.mkdir(parents=True, exist_ok=True)
    save_comparison_plot(timeseries, output_prefix)
    save_comparison_csv(timeseries, output_prefix)


if __name__ == "__main__":
    main()
