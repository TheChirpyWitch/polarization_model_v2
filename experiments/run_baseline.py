"""
Baseline experiment: single run driven by the weighted-PCA EIM signal.

Reproduces the main ICWSM experiment over ticks 699-1606
(2021-01-02 to 2023-12-25, weighted general-topic / general-narrative signal).

Usage
-----
    python -m experiments.run_baseline \
        --signal data/eim_1d_by_tick_v8_weighted_general_topic_general_narrative.json \
        --output outputs/baseline
"""

import argparse
from pathlib import Path

from polarization_model.model import EnhancedPolarizationModel
from polarization_model.parameters import ModelParameters
from polarization_model.signals import get_signal, load_json_signal
from polarization_model.visualization import export_data, plot_results

SIM_START = 699   # tick corresponding to 2021-01-02
SIM_END = 1606    # tick corresponding to 2023-12-25


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run baseline polarisation experiment.")
    parser.add_argument(
        "--signal",
        default="data/eim_1d_by_tick_v8_weighted_general_topic_general_narrative.json",
        help="Path to the weighted EIM JSON signal file.",
    )
    parser.add_argument(
        "--output",
        default="outputs/baseline",
        help="Output prefix for CSV and PNG artefacts.",
    )
    parser.add_argument("--num-people", type=int, default=50)
    parser.add_argument("--tau-base", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.seed is not None:
        import random, numpy as np
        random.seed(args.seed)
        np.random.seed(args.seed)

    print(f"Loading EIM signal from: {args.signal}")
    signal = load_json_signal(args.signal)
    print(f"  Tick range: {min(signal)} – {max(signal)}  ({len(signal)} ticks)")

    params = ModelParameters(
        num_people=args.num_people,
        tau_base=args.tau_base,
        tau_age_factor=0.003,
        tau_economic_factor=0.25,
        segregation_tolerance=0.25,
        rewire_prob=0.15,
        broadcast_interval=5,
    )

    model = EnhancedPolarizationModel(
        params=params,
        external_affect_fn=lambda tick: get_signal(signal, tick),
    )

    model.setup()
    model.tick = SIM_START  # fast-forward the model clock to the study period

    total_steps = SIM_END - SIM_START
    print(f"\nRunning {total_steps} ticks ({SIM_START} → {SIM_END})...")
    for i in range(total_steps):
        model.step()
        if (i + 1) % 200 == 0:
            print(f"  Step {i+1}/{total_steps}  |  model tick {model.tick}")

    print("\nFINAL STATE")
    model.print_status()
    model.print_agent_opinions()

    output_prefix = args.output
    Path(output_prefix).parent.mkdir(parents=True, exist_ok=True)
    export_data(model, prefix=output_prefix)
    plot_results(model, save_path=f"{output_prefix}_results.png")


if __name__ == "__main__":
    main()
