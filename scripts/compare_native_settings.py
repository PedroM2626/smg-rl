"""Compare fixed-action native workloads; none of these runs train a policy."""

import argparse
import json
from copy import deepcopy
from pathlib import Path

from smg_rl.benchmark import benchmark
from smg_rl.config import load_config


def compare(config_path, steps, output, repeats=2):
    original = load_config(config_path)
    results = []
    conditions = [(False, 1), (True, 1), (True, 8), (False, 8)]
    for repeat in range(repeats):
        for presentation, update_every in (conditions if repeat % 2 == 0 else conditions[::-1]):
            config = deepcopy(original)
            config["parallel"]["skip_present"] = presentation
            config["novelty"]["update_every"] = update_every
            folder = benchmark(config, steps)
            result = json.loads((folder / "benchmark.json").read_text(encoding="utf-8"))
            results.append({"repeat": repeat, "run": str(folder), **result})
            print(folder, flush=True)
    Path(output).write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--output", required=True)
    parser.add_argument("--repeats", type=int, default=2)
    args = parser.parse_args()
    compare(args.config, args.steps, args.output, args.repeats)
