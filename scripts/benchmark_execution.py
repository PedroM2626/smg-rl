"""Compare Dolphin execution settings without changing action frame counts."""

import argparse
import json
from copy import deepcopy
from pathlib import Path

from smg_rl.benchmark import benchmark
from smg_rl.config import load_config


def compare(config_path, steps, output, repeats=2, graphics=False):
    original = load_config(config_path)
    records = []
    for repeat in range(repeats):
        conditions = ["D3D", "Vulkan", "D3D12"] if graphics else [True, False]
        for condition in conditions if repeat % 2 == 0 else conditions[::-1]:
            config = deepcopy(original)
            key = "graphics_backend" if graphics else "cpu_thread"
            config["parallel"][key] = condition
            try:
                folder = benchmark(config, steps)
                record = json.loads((folder / "benchmark.json").read_text(encoding="utf-8"))
                record.update(run=str(folder), passed=True)
            except (RuntimeError, TimeoutError) as exc:
                record = {"passed": False, "error": str(exc)}
            records.append({"repeat": repeat, key: condition, **record})
            Path(output).write_text(json.dumps(records, indent=2), encoding="utf-8")
            print(condition, record.get("decisions_per_second_excluding_resets", record.get("error")), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--output", required=True)
    parser.add_argument("--graphics", action="store_true")
    args = parser.parse_args()
    compare(args.config, args.steps, args.output, args.repeats, args.graphics)
