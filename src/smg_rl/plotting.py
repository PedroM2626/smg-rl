"""Standalone, equal-scale top-down world X/Z trajectory plots."""

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection


def plot_trajectory(csv_path, output=None):
    csv_path = Path(csv_path)
    with csv_path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError("The trajectory contains no observations.")
    points = np.array([[float(row["x"]), float(row["z"])] for row in rows])
    fig, (ax, chart) = plt.subplots(1, 2, figsize=(11, 7), gridspec_kw={"width_ratios": [1, 1.1]})
    if len(points) > 1:
        segments = np.stack((points[:-1], points[1:]), axis=1)
        lines = LineCollection(segments, cmap="viridis", linewidth=2)
        lines.set_array(np.arange(len(segments)))
        lines.set_clim(0, max(1, len(points) - 1))
        ax.add_collection(lines)
        fig.colorbar(lines, ax=ax, label="Decision step", shrink=0.55)
    ax.scatter(*points[0], color="#1b9e77", marker="o", label="Start", zorder=3)
    ax.scatter(*points[-1], color="#d95f02", marker="X", label="Last observation", zorder=3)
    ax.autoscale()
    ax.set_aspect("equal", adjustable="datalim")
    ax.set(xlabel="World X", ylabel="World Z", title="Top-down trajectory (X/Z)")
    ax.legend(loc="best")
    progress = [100 * float(row["progress"]) for row in rows]
    chart.plot(progress, label="Current progress", alpha=0.65)
    best = (
        [100 * float(row["best_progress"]) for row in rows]
        if "best_progress" in rows[0]
        else np.maximum.accumulate(progress)
    )
    chart.plot(
        best,
        label="Reward-gated max projection" if "best_progress" in rows[0] else "Running max projection",
    )
    chart.set(
        xlabel="Decision step",
        ylabel="Course projection (%)",
        ylim=(0, max(1, min(100, 1.1 * max(max(progress), max(best))))),
        title="Progress toward the star",
    )
    chart.legend(loc="upper left", fontsize=9)
    chart.text(0.98, 0.02, "Full course = 100%", ha="right", transform=chart.transAxes, alpha=0.65)
    for axis in (ax, chart):
        axis.grid(alpha=0.2)
    fig.suptitle(f"{rows[0]['variation']} — {rows[-1]['outcome']}")
    fig.tight_layout()
    output = Path(output) if output else csv_path.with_suffix(".png")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    plt.close(fig)
    return output
