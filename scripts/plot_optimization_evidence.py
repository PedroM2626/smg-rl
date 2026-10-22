"""Render the retained engineering timings and counterfactual incentive audit."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    folder = Path(__file__).resolve().parent.parent / "docs/assets/performance"
    timings = json.loads((folder / "v2-benchmarks.json").read_text(encoding="utf-8"))
    audit = json.loads((folder / "v2-reward-audit.json").read_text(encoding="utf-8"))
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.3), layout="constrained")
    positions = np.arange(2)
    for index, (batch, label, color) in enumerate([(1, "Online", "#3979b6"), (8, "Batch of 8", "#e59631")]):
        values = []
        for game in ["smg2", "smg1"]:
            rows = [r["environment"]["curiosity"] for r in timings[game]["repeated"]
                    if not r["skip_present"] and r["curiosity_update_every"] == batch]
            values.append(sum(r["total_ms"] for r in rows) / sum(r["count"] for r in rows))
        bars = axes[0].bar(positions + (index - 0.5) * 0.35, values, width=0.35, label=label, color=color)
        axes[0].bar_label(bars, fmt="%.2f", padding=3)
    axes[0].set(title="Curiosity cost in live fixed-action runs", ylabel="Mean milliseconds per eligible sample")
    axes[0].legend()
    for index, (key, label, color) in enumerate([
        ("mean_recorded_return", "Recorded legacy return", "#3979b6"),
        ("mean_counterfactual_return", "Revised reward replay", "#e59631"),
    ]):
        values = [audit[g]["counterfactual"][key] for g in ["smg2", "smg1"]]
        bars = axes[1].bar(positions + (index - 0.5) * 0.35, values, width=0.35, label=label, color=color)
        axes[1].bar_label(bars, fmt="%.2f", padding=3)
    axes[1].set(title="Incentive audit on recorded failures", ylabel="Mean episode return (reward units)")
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].legend()
    for axis in axes:
        axis.set_xticks(positions, ["Galaxy 2", "Galaxy 1"])
        axis.margins(y=0.2)
        axis.spines[["top", "right"]].set_visible(False)
    figure.suptitle("Engineering measurements; reward replay is not a policy improvement result", fontsize=11)
    figure.savefig(folder / "v2-optimization.png", dpi=170)
    plt.close(figure)


if __name__ == "__main__":
    main()
