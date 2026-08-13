import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import re
import yaml
from datetime import datetime
from ops_utils.data.experiment import OpsDataset




def timing_plot(experiment: str) -> None:
    """
    Plot the time it takes to run each step of the pipeline (in minutes).
    Minimal robustness: skips malformed entries and handles simple units.
    """
    dataset = OpsDataset(experiment, method=None)
    timing_path = dataset.logfile
    try:
        with open(timing_path, "r") as file:
            timing_dict = yaml.safe_load(file) or {}
    except Exception as e:
        print(f"Could not read timing log at {timing_path}: {e}")
        return

    if not isinstance(timing_dict, dict) or not timing_dict:
        print("Timing log is empty or malformed. Skipping timing plot.")
        return

    func_names, time_minutes, timestamps = [], [], []
    for func_name, entry in timing_dict.items():
        # Try common fields first
        ran = None
        ts = None
        if isinstance(entry, dict):
            ran = entry.get("Ran in") or entry.get("ran_in") or entry.get("time")
            ts = entry.get("timestamp")

        minutes = None
        # Numeric directly
        if isinstance(ran, (int, float)):
            minutes = float(ran) / 60.0
        # Text with unit
        elif isinstance(ran, str):
            m = re.search(r"([0-9]*\.?[0-9]+)\s*([a-zA-Z]*)", ran)
            if m:
                val = float(m.group(1))
                unit = (m.group(2) or "s").lower()
                if unit.startswith("ms"):
                    minutes = val / 1000.0 / 60.0
                elif unit in {"", "s", "sec", "secs", "second", "seconds"}:
                    minutes = val / 60.0
                elif unit in {"m", "min", "mins", "minute", "minutes"}:
                    minutes = val
                elif unit in {"h", "hr", "hrs", "hour", "hours"}:
                    minutes = val * 60.0
        # Fallback numeric fields
        if minutes is None and isinstance(entry, dict):
            for k in (
                "duration_min",
                "duration",
                "duration_s",
                "time_seconds",
                "seconds",
            ):
                if k in entry:
                    try:
                        val = float(entry[k])
                        minutes = val if "min" in k else val / 60.0
                        break
                    except Exception:
                        pass

        if minutes is None:
            continue

        # Parse timestamp lightly (single common format)
        dt = None
        if isinstance(ts, str):
            try:
                dt = datetime.strptime(ts, "%Y-%m-%d %H:%M:%S")
            except Exception:
                dt = None

        func_names.append(func_name)
        time_minutes.append(minutes)
        timestamps.append(dt)

    if not time_minutes:
        print("No valid timing entries found. Skipping timing plot.")
        return

    # Sort by timestamp when available
    order = sorted(
        range(len(func_names)),
        key=lambda i: (timestamps[i] is None, timestamps[i] or datetime.max),
    )
    func_names = [func_names[i] for i in order]
    time_minutes = np.array([time_minutes[i] for i in order], dtype=float)
    time_hours = time_minutes / 60.0
    cumulative_hours = np.cumsum(time_hours)

    # Scale figure width based on number of steps (min 10, ~0.8 inches per step)
    fig_width = max(10, len(func_names) * 0.8)
    fig, ax1 = plt.subplots(figsize=(fig_width, 6))
    ax1.bar(func_names, time_hours, label="Step Time", alpha=0.7)
    ax1.set_ylabel("Time (hours)")
    ax1.set_title("Pipeline timing")
    plt.xticks(rotation=45, ha="right")

    ax2 = ax1.twinx()
    ax2.plot(
        func_names, cumulative_hours, color="red", marker="o", label="Cumulative Time"
    )
    ax2.set_ylabel("Cumulative Time (hours)")

    # Legend combining
    l1, lab1 = ax1.get_legend_handles_labels()
    l2, lab2 = ax2.get_legend_handles_labels()
    ax1.legend(l1 + l2, lab1 + lab2, loc="upper left")

    plt.tight_layout()
    fig.suptitle("Pipeline timing")
    ax1.set_title(experiment, fontsize=10)
    try:
        plt.savefig(dataset.metrics_paths["timing"], dpi=300)
    except Exception as e:
        print(f"Failed to save timing plot: {e}")

    return