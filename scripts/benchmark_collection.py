"""Measure fixed-action synchronous worker idle time and observation transport."""

import argparse
import json
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch

from smg_rl.config import load_config
from smg_rl.parallel import WorkerFactory
from smg_rl.train import new_run
from smg_rl.vector import MeasuredSubprocVecEnv


def benchmark(config_path, steps, workers, output, repeats=2):
    original = load_config(config_path)
    torch.set_num_threads(original["training"]["torch_threads"])
    results = []
    conditions = [(False, "ordered"), (False, "ready"), (True, "ready"), (True, "ordered")]
    for repeat in range(repeats):
        for shared, order in conditions if repeat % 2 == 0 else conditions[::-1]:
            config = deepcopy(original)
            config["parallel"].update(workers=workers, shared_observations=shared,
                                      receive_order=order, collector="measured")
            folder = new_run(config, "benchmark")
            config["paths"]["user"] = str(Path(config["root"]) / ".local/collection-bench" / folder.name / "template")
            shape = (config["environment"]["frame_stack"], config["environment"]["image_size"],
                     config["environment"]["image_size"])
            env = MeasuredSubprocVecEnv([WorkerFactory(config, i, str(folder)) for i in range(workers)],
                                       shape, shared=shared, receive_order=order)
            durations, idle, compute, bottlenecks = [], [], [], []
            try:
                env.seed(42)
                env.reset()
                for index in range(steps):
                    started = time.perf_counter()
                    env.step(np.full(workers, [0, 1, 9, 3, 10][index % 5]))
                    durations.append(time.perf_counter() - started)
                    idle.append(env.last_statistics["barrier_idle_ms"])
                    compute.append(env.last_statistics["compute_ms"])
                    bottlenecks.append(env.last_statistics["slowest_worker"])
                result = {"run": str(folder), "repeat": repeat, "workers": workers,
                          "vector_steps": steps, "aggregate_decisions": steps * workers,
                          "wall_seconds": sum(durations), "aggregate_decisions_per_second": steps * workers / sum(durations),
                          "shared_observations": shared, "receive_order": order,
                          "mean_worker_compute_ms": np.mean(compute, axis=0).tolist(),
                          "mean_worker_barrier_idle_ms": np.mean(idle, axis=0).tolist(),
                          "bottleneck_count_per_worker": [bottlenecks.count(i) for i in range(workers)],
                          "collector": env.timing_snapshot(),
                          "note": "Fixed actions, no policy/optimization. Reset time included in vector-step wall time. Synchronous barrier retained."}
                results.append(result)
                (folder / "collection.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
                Path(output).write_text(json.dumps(results, indent=2), encoding="utf-8")
                print(shared, order, round(result["aggregate_decisions_per_second"], 3), folder, flush=True)
            finally:
                env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    benchmark(args.config, args.steps, args.workers, args.output, args.repeats)
