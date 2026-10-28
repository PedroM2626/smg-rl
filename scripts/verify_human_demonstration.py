"""Replay a recorded human demonstration in a private frame-stepped Dolphin."""

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np

from smg_rl.checkpoints import digest
from smg_rl.config import load_config
from smg_rl.diagnostics import atomic_json
from smg_rl.human_recorder import (
    FRAMES_PER_SAMPLE,
    GAME_ID,
    confirmed_star,
    controls_mask,
    exact_observation,
    start_is_canonical,
)
from smg_rl.native import NativeDolphin
from smg_rl.parallel import runtime_directory
from smg_rl.train import new_run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('demonstration', type=Path)
    parser.add_argument('--max-seconds', type=int, default=900)
    args = parser.parse_args()
    if not 1 <= args.max_seconds <= 1800:
        raise ValueError('The replay deadline must be between 1 and 1800 seconds.')
    source = args.demonstration.resolve()
    manifest = json.loads((source / 'manifest.json').read_text(encoding='utf-8'))
    if (manifest['state'] != 'completed' or manifest['game_id'] != GAME_ID
            or manifest['scenario'] != 1 or manifest['start_slot'] != 1
            or not manifest['full_start_ram_star']
            or manifest['emulated_frames_per_sample'] != FRAMES_PER_SAMPLE):
        raise ValueError('The manifest is not a full-start two-frame SMG1 demonstration.')
    for name, expected in manifest['files_sha256'].items():
        if Path(name).name != name or digest(source / name) != expected:
            raise ValueError(f'Demonstration file integrity check failed: {name}')
    with (source / 'inputs.jsonl').open(encoding='utf-8') as stream:
        inputs = [json.loads(line) for line in stream]
    with (source / 'trajectory.csv').open(encoding='utf-8', newline='') as stream:
        rows = list(csv.DictReader(stream))
    n = manifest['samples']
    if len(inputs) != n or len(rows) != n + 1 or n <= 0:
        raise ValueError('Input or trajectory length differs from the manifest.')
    if (source / 'observations.gray8').stat().st_size != (n + 1) * 84 * 84:
        raise ValueError('Observation length differs from the input timeline.')
    for i, item in enumerate(inputs):
        if (item['sample'] != i or item['emulated_frame_start'] != i * FRAMES_PER_SAMPLE
                or item['duration_frames'] != FRAMES_PER_SAMPLE
                or item['mask'] != controls_mask(item['controls'])):
            raise ValueError(f'Invalid control timeline at sample {i}.')
    config = load_config('config/smg1-six-two-hour.yaml')
    config['paths']['runs'] = str(Path(config['root']) / 'runs/human-verification')
    config['parallel'].update(frame_skip=FRAMES_PER_SAMPLE, native_capture='full', workers=1)
    config['replay']['enabled'] = False
    state = Path(config['paths']['assets']) / f'{GAME_ID}.s01'
    if digest(state) != manifest['source_state_sha256']:
        raise ValueError('The current starting savestate differs from the recorded source.')
    run = new_run(config, 'human-replay')
    config['paths']['user'] = str(runtime_directory(config, run) / 'template')
    result = {'state': 'running', 'source': str(source), 'run': str(run),
              'recorded_samples': n, 'real_replay_samples': 0, 'ram_star': False,
              'first_position_divergence': None, 'first_observation_divergence': None,
              'matching_observations': 0, 'max_position_error': 0.0}
    atomic_json(run / 'result.json', result)
    print(json.dumps({'run': str(run)}), flush=True)
    observations = np.memmap(source / 'observations.gray8', dtype=np.uint8,
                             mode='r', shape=(n + 1, 84, 84))
    backend = NativeDolphin(config)
    try:
        backend.load_slot(1)
        initial = backend.memory.sample()
        if not start_is_canonical(initial, config):
            raise ValueError('Replay did not start at the canonical living state.')
        deadline = time.monotonic() + args.max_seconds
        for i in range(n + 1):
            if time.monotonic() >= deadline:
                result['state'] = 'deadline'
                break
            if i:
                backend.advance(tuple(inputs[i - 1]['controls']))
            t = backend.memory.sample()
            expected_position = np.array([float(rows[i][k]) for k in ('x', 'y', 'z')])
            error = float(np.linalg.norm(np.array(t.position) - expected_position))
            result['max_position_error'] = max(result['max_position_error'], error)
            if error > 0.1 and result['first_position_divergence'] is None:
                result['first_position_divergence'] = {'sample': i, 'error': error}
            if np.array_equal(exact_observation(backend), observations[i]):
                result['matching_observations'] += 1
            elif result['first_observation_divergence'] is None:
                result['first_observation_divergence'] = i
            result['real_replay_samples'] = i
            if t.dead or t.complete:
                result['ram_star'] = confirmed_star(t, True, i)
                result['terminal'] = t.to_dict()
                result['state'] = 'star' if result['ram_star'] else 'terminal_without_star'
                break
            if i and i % 200 == 0:
                atomic_json(run / 'result.json', result)
        else:
            result['state'] = 'inputs_exhausted_without_terminal'
        result['source_savestate_unchanged'] = digest(state) == manifest['source_state_sha256']
    except BaseException as exc:
        result.update(state='failed', error=repr(exc))
        raise
    finally:
        backend.close()
        atomic_json(run / 'result.json', result)
        print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
