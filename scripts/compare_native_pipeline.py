"""Fixed-action event/capture comparisons on one protocol-v3 build."""

import argparse
import json
from copy import deepcopy
from pathlib import Path

from smg_rl.benchmark import benchmark
from smg_rl.config import load_config


def compare(config_path, steps, repeats, output):
    original = load_config(config_path)
    results = []
    conditions = [("poll", "full", False), ("events", "full", False),
                  ("events", "cpu", False), ("events", "cpu", True)]
    for repeat in range(repeats):
        for wakeup, capture, quiet in conditions if repeat % 2 == 0 else conditions[::-1]:
            config = deepcopy(original)
            config["parallel"].update(bridge_wakeup=wakeup, native_capture=capture, quiet_ui=quiet)
            folder = benchmark(config, steps)
            result = json.loads((folder / "benchmark.json").read_text(encoding="utf-8"))
            results.append({"repeat": repeat, "run": str(folder), **result})
            Path(output).write_text(json.dumps(results, indent=2), encoding="utf-8")
            print(wakeup, capture, quiet, round(result["decisions_per_second_excluding_resets"], 3), folder, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    compare(args.config, args.steps, args.repeats, args.output)
