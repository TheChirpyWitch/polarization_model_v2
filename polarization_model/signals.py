"""
EIM / NEIM signal loaders and accessor utilities.

Tick-indexed signals are used to drive the external affect component of
Agent Zero++.  Two source formats are supported:

  - JSON  : weighted PCA signal  (eim_1d_by_tick_*.json)
  - Excel : per-source NEIM signals (neim_1d_by_tick_*.xlsx), each with
            columns [Tick, NEIM_1D_Value]

All loaders return a {int tick -> float value} dict.
The get_signal() accessor clamps out-of-range ticks to the boundary values.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def load_json_signal(filepath: str | Path) -> dict[int, float]:
    """Load a weighted-PCA EIM signal from a JSON file."""
    with open(filepath, "r") as f:
        raw = json.load(f)
    return {int(k): float(v) for k, v in raw.items()}


def load_excel_signal(filepath: str | Path) -> dict[int, float]:
    """Load a per-source NEIM signal from an Excel file (Tick, NEIM_1D_Value)."""
    df = pd.read_excel(filepath)
    return {int(row["Tick"]): float(row["NEIM_1D_Value"]) for _, row in df.iterrows()}


def get_signal(signal: dict[int, float], tick: int) -> float:
    """
    Return the signal value at the requested tick.

    Clamps to the nearest boundary value when tick is outside the observed range,
    so simulations can safely request any tick without a KeyError.
    """
    min_tick = min(signal)
    max_tick = max(signal)
    tick = max(min_tick, min(tick, max_tick))
    return signal[tick]
