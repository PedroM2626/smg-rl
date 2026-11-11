"""Compare Dolphin thread options on one worker: decisions per second without resets, and the cost of a reset.

Conditions alternate in reverse order on every repeat. Run it with nothing else using the CPU, or report the load.

Usage: python scripts/benchmark_emulator_options.py CONFIG OUTPUT.json [--steps 96] [--resets 6] [--repeats 2]
"""
import argparse
import json
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import psutil

from smg_rl.actions import action_set
from smg_rl.config import load_config
from smg_rl.env import GalaxyEnv
from smg_rl.native import background_backend
from smg_rl.train import new_run

CONDITIONS = {'baseline': {}, 'cpu_thread_off': {'cpu_thread': False}, 'dsp_thread_off': {'dsp_thread': False},
              'both_off': {'cpu_thread': False, 'dsp_thread': False}}


def run(config, steps, resets):
    config = deepcopy(config)
    config['parallel']['workers'] = 1
    output = new_run(config, 'benchmark')
    config['paths']['user'] = str(Path(config['root']) / '.local' / 'emulator-options' / output.name / 'template')
    config['paths']['runs'] = str(output.parent)
    backend = background_backend(config)
    env = GalaxyEnv(config, backend, output / 'worker-00', memory=backend.memory, variation=0, intrinsic=False)
    try:
        names = action_set(config)
        actions = [names.index(n) for n in ((), ('w',), ('jump',), ('s',), ('w', 'jump')) if n in names]
        env.reset(seed=42)
        load_before = psutil.cpu_percent(interval=None)
        started, done, step_times = time.perf_counter(), 0, []
        while done < steps:
            t = time.perf_counter()
            _, _, terminated, truncated, _ = env.step(actions[done % len(actions)])
            step_times.append(time.perf_counter() - t)
            done += 1
            if terminated or truncated:
                env.reset(seed=42)
        total = float(sum(step_times))
        load = psutil.cpu_percent(interval=None)
        reset_times = []
        for _ in range(resets):
            t = time.perf_counter()
            env.reset(seed=42)
            reset_times.append(time.perf_counter() - t)
        del started, load_before
        return {'decisions_per_second': steps / total, 'median_step_ms': 1000 * float(np.median(step_times)),
                'reset_median_s': float(np.median(reset_times)), 'reset_max_s': float(max(reset_times)),
                'system_cpu_percent_during_steps': load}
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config')
    parser.add_argument('output', type=Path)
    parser.add_argument('--steps', type=int, default=96)
    parser.add_argument('--resets', type=int, default=6)
    parser.add_argument('--repeats', type=int, default=2)
    args = parser.parse_args()
    original = load_config(args.config)
    original['replay']['enabled'] = False
    records = []
    names = list(CONDITIONS)
    for repeat in range(args.repeats):
        for name in names if repeat % 2 == 0 else names[::-1]:
            config = deepcopy(original)
            config['parallel'].update(CONDITIONS[name])
            psutil.cpu_percent(interval=None)
            try:
                record = run(config, args.steps, args.resets)
            except Exception as error:  # a rejected option is a result, not a crash
                record = {'error': repr(error)[:300]}
            records.append({'repeat': repeat, 'condition': name, **record})
            args.output.write_text(json.dumps(records, indent=1))
            print(name, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in record.items()}, flush=True)


if __name__ == '__main__':
    main()
