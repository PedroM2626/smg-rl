"""Plot retained pipeline and synchronous collection comparisons without gameplay assets."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    folder = Path(__file__).resolve().parent.parent / "docs/assets/performance"
    pipelines = json.loads((folder / "v3-pipeline-final.json").read_text(encoding="utf-8"))
    collection = json.loads((folder / "v3-collection-final.json").read_text(encoding="utf-8"))
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.7), layout="constrained")
    conditions = [("poll", "full", False), ("events", "full", False),
                  ("events", "cpu", False), ("events", "cpu", True)]
    for index, game in enumerate(["smg2", "smg1"]):
        rates = []
        for condition in conditions:
            rows = [r for r in pipelines[game] if
                    (r["bridge_wakeup"], r["native_capture"], r["quiet_ui"]) == condition]
            rates.append(sum(r["steps"] for r in rows) / sum(r["step_wall_seconds"] for r in rows))
        bars = axes[0].bar(np.arange(4) + (index - 0.5) * 0.36, rates,
                           width=0.36, label="Galaxy 2" if index == 0 else "Galaxy 1")
        axes[0].bar_label(bars, fmt="%.1f", padding=3)
    axes[0].set_xticks(np.arange(4), ["Poll / full", "Events / full", "Events / CPU", "Events / CPU\nQuiet UI"])
    axes[0].set(title="One-worker environment; resets and PPO excluded", ylabel="Decisions per second")
    axes[0].legend(loc="upper left")
    rates = []
    for shared, order in [(False, "ordered"), (False, "ready"), (True, "ready"), (True, "ordered")]:
        rows = [r for r in collection if r["shared_observations"] == shared and r["receive_order"] == order]
        rates.append(sum(r["aggregate_decisions"] for r in rows) / sum(r["wall_seconds"] for r in rows))
    bars = axes[1].bar(np.arange(4), rates, color=["#3979b6", "#8a9cad", "#8a9cad", "#8a9cad"])
    axes[1].bar_label(bars, fmt="%.1f", padding=3)
    axes[1].set_xticks(np.arange(4), ["Ordered\npipes", "Ready\npipes", "Ready\nshared", "Ordered\nshared"])
    axes[1].set(title="Four-worker Galaxy 2 collection; PPO excluded", ylabel="Aggregate decisions per second")
    for axis in axes:
        axis.margins(y=0.28)
        axis.spines[["top", "right"]].set_visible(False)
        axis.tick_params(axis="x", labelsize=9)
    figure.suptitle("Protocol v3 engineering comparisons: two passes in reversed order", fontsize=12)
    figure.savefig(folder / "v3-pipeline.png", dpi=170)
    plt.close(figure)


if __name__ == "__main__":
    main()
