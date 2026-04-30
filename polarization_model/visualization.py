"""
Visualisation and export utilities.

plot_results()   — 3×3 diagnostic panel covering opinions, disposition,
                   agent states, network homophily, economic metrics, and
                   spatial distribution.

export_data()    — write the population-level summary to CSV.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pandas as pd


def plot_results(model, save_path: str | Path | None = None) -> plt.Figure:
    """
    Generate a 3×3 diagnostic panel for a completed model run.

    Parameters
    ----------
    model     : EnhancedPolarizationModel after running.
    save_path : If provided, save the figure to this path (PNG recommended).
    """
    fig, axes = plt.subplots(3, 3, figsize=(14, 11))

    # 1. Individual opinion trajectories
    ax = axes[0, 0]
    for aid in model.data.agent_opinions:
        ax.plot(model.data.ticks, model.data.agent_opinions[aid],
                alpha=0.3, linewidth=0.5)
    ax.set_ylim(-1, 1)
    ax.axhline(0, color="black", linestyle="--", alpha=0.3)
    ax.set_xlabel("Ticks")
    ax.set_ylabel("Opinion")
    ax.set_title("Opinions Over Time")

    # 2. Population disposition (mean and max)
    ax = axes[0, 1]
    ax.plot(model.data.ticks, model.data.disposition_mean, label="Mean", color="blue")
    ax.plot(model.data.ticks, model.data.disposition_max, label="Max", color="red", alpha=0.7)
    ax.axhline(0, color="red", linestyle="--", label="Threshold")
    ax.set_xlabel("Ticks")
    ax.set_ylabel("Disposition")
    ax.set_title("Disposition Over Time")
    ax.legend(fontsize=8)

    # 3. Behavioural-state stacked area
    ax = axes[0, 2]
    ax.stackplot(
        model.data.ticks,
        model.data.count_quiet,
        model.data.count_agitated,
        model.data.count_flight,
        model.data.count_protest,
        model.data.count_riot,
        model.data.count_mob,
        labels=["Quiet", "Agitated", "Flight", "Protest", "Riot", "Mob"],
        colors=["#3B6D11", "#DAA520", "#185FA5", "#E69500", "#D85A30", "#A32D2D"],
        alpha=0.7,
    )
    ax.set_xlabel("Ticks")
    ax.set_ylabel("Count")
    ax.set_title("Agent States")
    ax.legend(loc="upper left", fontsize=8)

    # 4. Network homophily (echo-chamber proxy)
    ax = axes[1, 0]
    ax.plot(model.data.ticks, model.data.network_homophily, color="purple")
    ax.set_ylim(0, 1)
    ax.set_xlabel("Ticks")
    ax.set_ylabel("Homophily")
    ax.set_title("Network Homophily (Echo Chamber)")

    # 5. Cost of goods (inflation trajectory)
    ax = axes[1, 1]
    ax.plot(model.data.ticks, model.data.cost_of_goods, color="green")
    ax.set_xlabel("Ticks")
    ax.set_ylabel("Cost ($)")
    ax.set_title("Cost of Goods")

    # 6. Employment and housing
    ax = axes[1, 2]
    ax.plot(model.data.ticks, model.data.count_employed, label="Employed", color="green")
    ax.plot(model.data.ticks, model.data.count_unhoused, label="Unhoused", color="red")
    ax.set_xlabel("Ticks")
    ax.set_ylabel("Count")
    ax.set_title("Economic Status")
    ax.legend(fontsize=8)

    # 7. Final opinion distribution
    ax = axes[2, 0]
    final_opinions = [model.data.agent_opinions[i][-1] for i in range(len(model.people))]
    ax.hist(final_opinions, bins=20, range=(-1, 1), edgecolor="black", alpha=0.7)
    ax.set_xlabel("Opinion")
    ax.set_ylabel("Count")
    ax.set_title("Final Opinion Distribution")

    # 8. Final disposition distribution
    ax = axes[2, 1]
    final_disp = [model.data.agent_dispositions[i][-1] for i in range(len(model.people))]
    ax.hist(final_disp, bins=20, edgecolor="black", alpha=0.7, color="orange")
    ax.axvline(0, color="red", linestyle="--", label="Threshold")
    ax.set_xlabel("Disposition")
    ax.set_ylabel("Count")
    ax.set_title("Final Disposition Distribution")
    ax.legend(fontsize=8)

    # 9. Final spatial distribution coloured by behavioural state
    ax = axes[2, 2]
    for p in model.people:
        ax.scatter(p.x, p.y, c=p.get_color(), s=30, alpha=0.7)
    ax.set_xlim(model.min_coord, model.max_coord)
    ax.set_ylim(model.min_coord, model.max_coord)
    ax.set_aspect("equal")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title("Spatial Distribution")

    color_map = {
        "Quiet":    "#3B6D11",
        "Agitated": "#DAA520",
        "Flight":   "#185FA5",
        "Protest":  "#E69500",
        "Riot":     "#D85A30",
        "Mob":      "#A32D2D",
        "Unhoused": "#888780",
    }
    patches = [mpatches.Patch(color=c, label=s) for s, c in color_map.items()]
    ax.legend(handles=patches, fontsize=7, loc="upper right")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved figure: {save_path}")

    return fig


def export_data(model, prefix: str = "enhanced_polarization") -> pd.DataFrame:
    """Write the population-level summary DataFrame to CSV and return it."""
    df = model.data.get_summary_df()
    path = f"{prefix}_summary.csv"
    df.to_csv(path, index=False)
    print(f"Saved data: {path}")
    return df
