"""Replay a verified human star and validate immutable grounded intermediate states."""

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import psutil

from smg_rl.checkpoints import digest
from smg_rl.config import load_config
from smg_rl.diagnostics import atomic_json
from smg_rl.human_recorder import confirmed_star, exact_observation
from smg_rl.native import NativeDolphin
from smg_rl.resource_memory import memory_snapshot


def prepare(source, route_path, output, maximum_seconds=1200):
    source, route_path, output = map(Path, (source, route_path, output))
    if not 60 <= maximum_seconds <= 1800 or output.exists():
        raise ValueError('Use a new output directory and a finite preparation budget.')
    if any((p.info['name'] or '').lower() == 'dolphin.exe' for p in psutil.process_iter(['name'])):
        raise ValueError('Existing emulators must finish before preparation.')
    manifest = json.loads((source / 'manifest.json').read_text())
    route = json.loads(route_path.read_text())
    if (manifest['game_id'] != 'RMGE01' or manifest['scenario'] != 1
            or manifest['state'] != 'completed' or not manifest['full_start_ram_star']
            or manifest['emulated_frames_per_sample'] != 2 or route['schema'] != 5):
        raise ValueError('A full-start SMG1 star recording and schema5 route are required.')
    for name, expected in manifest['files_sha256'].items():
        if Path(name).name != name or digest(source / name) != expected:
            raise ValueError('Human source integrity failed.')
    if digest(source / 'inputs.jsonl') != route['source_inputs_sha256']:
        raise ValueError('Route belongs to another human recording.')
    with (source / 'trajectory.csv').open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    points = np.array([[float(r[k]) for k in ('x', 'y', 'z')] for r in rows])
    indices = {}
    for waypoint in (1, 2, 4, 8, 14, 18, 23, 29, 34, 38, 40):
        if waypoint >= len(route['waypoints']) - 1:
            continue
        error = np.linalg.norm(points - np.array(route['waypoints'][waypoint]), axis=1)
        index = int(error.argmin())
        if error[index] > .01 or rows[index]['grounded'] != 'True':
            raise ValueError('Human waypoint is not an exact grounded recorded sample.')
        indices[index] = waypoint
    inputs = [json.loads(line) for line in (source / 'inputs.jsonl').read_text().splitlines()]
    images = np.memmap(source / 'observations.gray8', dtype=np.uint8, mode='r',
                       shape=(len(rows), 84, 84))
    output.mkdir(parents=True)
    config = load_config('config/iqn-day-smg1-20261029.yaml')
    if digest(Path(config['paths']['assets']) / 'RMGE01.s01') != manifest['source_state_sha256']:
        raise ValueError('Canonical source savestate changed.')
    config['paths']['user'] = str((output / 'profile' / 'template').resolve())
    config['parallel'].update(frame_skip=2, native_capture='full', workers=1)
    calibration = json.loads(Path(config['paths']['calibration']).read_text())['variations']['sweet_sweet']
    goal = np.array(calibration['goal'])
    report = {'state': 'running', 'source': str(source.resolve()),
              'source_manifest_sha256': digest(source / 'manifest.json'),
              'route_sha256': digest(route_path), 'recorded_samples': len(inputs),
              'human_replay_samples': 0, 'idle_probe_frames': 0, 'human_ram_star': False,
              'matching_observations': 0, 'candidates': [], 'max_seconds': maximum_seconds,
              'scope': 'Human replay and engineering probes only, no learner or agent success.'}
    backend = None
    deadline = time.monotonic() + maximum_seconds
    try:
        backend = NativeDolphin(config)
        backend.load_slot(1)
        for index in range(len(rows)):
            if time.monotonic() >= deadline or memory_snapshot()['physical_available_bytes'] < 3 * 2**30:
                raise TimeoutError('Preparation time or memory boundary reached.')
            if index:
                backend.advance(tuple(inputs[index - 1]['controls']))
                report['human_replay_samples'] += 1
            t = backend.memory.sample()
            if np.linalg.norm(np.array(t.position) - points[index]) > .1:
                raise ValueError(f'Human replay position diverged at{index}.')
            if not np.array_equal(exact_observation(backend), images[index]):
                raise ValueError(f'Human replay observation diverged at{index}.')
            report['matching_observations'] += 1
            if index in indices:
                waypoint = indices[index]
                name = f'human-waypoint-{waypoint:02d}.state'
                path = output / name
                backend.save_state(path)
                report['candidates'].append({'file': name, 'name': f'human-waypoint-{waypoint:02d}',
                    'sha256': digest(path), 'position': list(t.position), 'health': t.health,
                    'scene_frame': t.scene_frame, 'scenario': 1, 'origin_slot': 1,
                    'near_star_fixture': False, 'route_waypoint': waypoint, 'region': waypoint,
                    'route_sha256': digest(route_path), 'human_sample': index, 'accepted': False,
                    'validation': {'passed': False, 'idle_frames': 60, 'repetitions': 2}})
                atomic_json(output / 'preparation.json', report)
            if t.dead:
                raise ValueError('Human replay died before its recorded star.')
            if t.complete:
                report['human_ram_star'] = confirmed_star(t, True, index, 'RMGE01', 1)
                break
        if not report['human_ram_star'] or report['human_replay_samples'] != len(inputs):
            raise ValueError('Exact full-start human completion was not reproduced.')
        for candidate in report['candidates']:
            if np.linalg.norm(np.array(candidate['position']) - goal) < 2000:
                candidate['rejection'] = 'Too close to the retained near-star reference.'
                continue
            passed = True
            for repetition in range(2):
                backend.load_state(output / candidate['file'])
                restored = backend.memory.sample()
                exact = (restored.health == candidate['health'] and restored.scene_frame == candidate['scene_frame']
                         and restored.grounded is True and not restored.dead and not restored.complete
                         and np.linalg.norm(np.array(restored.position) - candidate['position']) <= .01)
                passed &= exact
                for _ in range(30):
                    if time.monotonic() >= deadline:
                        raise TimeoutError('Idle verification deadline.')
                    backend.advance(())
                    report['idle_probe_frames'] += 2
                    t = backend.memory.sample()
                    passed &= not t.dead and not t.complete and t.grounded is True and t.health == candidate['health']
                candidate['validation'][f'probe_{repetition}_end'] = t.to_dict()
            backend.load_state(output / candidate['file'])
            restored = backend.memory.sample()
            passed &= restored.scene_frame == candidate['scene_frame'] and restored.health == candidate['health']
            passed &= np.linalg.norm(np.array(restored.position) - candidate['position']) <= .01
            candidate['accepted'] = candidate['validation']['passed'] = bool(passed)
            if not passed:
                candidate['rejection'] = 'Grounded/full-health idle or exact reset check failed.'
            atomic_json(output / 'preparation.json', report)
        entries = sorted([c for c in report['candidates'] if c['accepted']], key=lambda c: c['route_waypoint'])
        if len(entries) < 3 or entries[-1]['route_waypoint'] < len(route['waypoints']) // 2:
            raise ValueError('Insufficient safe anchors covering later human-route regions.')
        bank = {'schema': 2, 'game_id': 'RMGE01', 'emulator_sha256': digest(config['paths']['dolphin']),
                'route_sha256': digest(route_path), 'source_full_start_ram_star': True,
                'source_manifest_sha256': report['source_manifest_sha256'], 'entries': entries,
                'scope': 'Validated human-assisted starts, not learned mission completions.'}
        atomic_json(output / 'bank.json', bank)
        report.update(state='completed', accepted=len(entries), bank_sha256=digest(output / 'bank.json'))
    except BaseException as exc:
        report.update(state='failed', error=repr(exc))
        raise
    finally:
        if backend is not None:
            backend.close()
        atomic_json(output / 'preparation.json', report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('--route', type=Path, default=Path('config/routes/smg1-human-landing-v2.json'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-seconds', type=int, default=1200)
    args = parser.parse_args()
    print(json.dumps(prepare(args.source, args.route, args.output, args.max_seconds), indent=2))
