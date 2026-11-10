"""Replay a verified full-start human star and publish validated start anchors for any configured mission.

Usage: python scripts/prepare_anchor_bank.py SOURCE ROUTE CONFIG VARIATION OUTPUT [--count 11]

The anchors are evenly spaced along the recorded path (by 3D arc length), grounded, at least 2000 units
from the goal, and must survive an exact reload plus a 60-frame idle probe twice. The bank is a set of
spawn points for training, never a demonstration: no human action is ever fed to a learner.
"""
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
from smg_rl.human_recorder import (
    FRAMES_PER_SAMPLE,
    confirmed_star,
    exact_observation,
    start_is_canonical,
)
from smg_rl.native import NativeDolphin
from smg_rl.resource_memory import memory_snapshot


def choose_samples(points, grounded, goal, count):
    """Pick one grounded sample near each of `count` evenly spaced arc-length fractions."""
    arc = np.concatenate([[0.], np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))])
    chosen = []
    for k in range(1, count + 1):
        target = arc[-1] * k / (count + 1)
        order = np.argsort(np.abs(arc - target))
        for index in order[:200]:
            # Stay grounded for the next 30 samples of the recording so a 60-frame idle probe is plausible.
            if (grounded[index] and index not in chosen and grounded[index:index + 30].all()
                    and np.linalg.norm(points[index] - goal) >= 2000):
                chosen.append(int(index))
                break
    return sorted(chosen)


def nearest_waypoints(route, points):
    waypoints = np.asarray(route['waypoints'], dtype=float)
    return [int(np.linalg.norm(waypoints - p, axis=1).argmin()) for p in points]


def prepare(source, route_path, config_path, variation, output, count, maximum_seconds=1500, goal_candidate=False):
    source, route_path, output = map(Path, (source, route_path, output))
    if not 60 <= maximum_seconds <= 1800 or output.exists():
        raise ValueError('Use a new output directory and a finite preparation budget.')
    if any((p.info['name'] or '').lower() == 'dolphin.exe' for p in psutil.process_iter(['name'])):
        raise ValueError('Existing emulators must finish before preparation.')
    manifest = json.loads((source / 'manifest.json').read_text())
    route = json.loads(route_path.read_text())
    expected_state = 'candidate_goal' if goal_candidate else 'completed'
    if (manifest['state'] != expected_state or bool(manifest.get('full_start_ram_star')) == goal_candidate
            or manifest['emulated_frames_per_sample'] != FRAMES_PER_SAMPLE or route['schema'] != 6):
        raise ValueError('A completed full-start star recording (or, with goal_candidate, an owner-marked goal '
                         'recording) and a schema 6 route are required.')
    for name, expected in manifest['files_sha256'].items():
        if Path(name).name != name or digest(source / name) != expected:
            raise ValueError('Human source integrity failed.')
    config = load_config(config_path)
    mission = next(v for v in config['variations'] if v['name'] == variation)
    game_id = config['game']['id']
    if manifest['game_id'] != game_id or manifest['scenario'] != mission['scenario']:
        raise ValueError('The recording belongs to another game or mission.')
    state_file = Path(config['paths']['assets']) / f'{game_id}.s{mission["start_slot"]:02d}'
    if digest(state_file) != manifest['source_state_sha256']:
        raise ValueError('Canonical source savestate changed.')
    with (source / 'trajectory.csv').open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    points = np.array([[float(r[k]) for k in ('x', 'y', 'z')] for r in rows])
    grounded = np.array([r['grounded'] == 'True' for r in rows])
    calibration = json.loads(Path(config['paths']['calibration']).read_text())['variations'][variation]
    goal = np.array(calibration['goal'])
    samples = choose_samples(points, grounded, goal, count)
    waypoints = nearest_waypoints(route, points[samples])
    # Keep strictly increasing route indices of at least 1, as the backward curriculum requires.
    selected, last = [], 0
    for sample, waypoint in zip(samples, waypoints):
        if waypoint > last:
            selected.append((sample, waypoint))
            last = waypoint
    inputs = [json.loads(line) for line in (source / 'inputs.jsonl').read_text().splitlines()]
    images = np.memmap(source / 'observations.gray8', dtype=np.uint8, mode='r', shape=(len(rows), 84, 84))
    output.mkdir(parents=True)
    config['paths']['user'] = str((output / 'profile' / 'template').resolve())
    config['parallel'].update(frame_skip=FRAMES_PER_SAMPLE, native_capture='full', workers=1)
    wanted = {sample: waypoint for sample, waypoint in selected}
    report = {'state': 'running', 'source': str(source.resolve()), 'game_id': game_id, 'variation': variation,
              'source_manifest_sha256': digest(source / 'manifest.json'), 'route_sha256': digest(route_path),
              'recorded_samples': len(inputs), 'human_replay_samples': 0, 'matching_observations': 0,
              'human_ram_star': False, 'candidates': [], 'max_seconds': maximum_seconds,
              'scope': 'Replay and spawn-point validation only; no learner or agent success.'}
    backend = None
    deadline = time.monotonic() + maximum_seconds
    try:
        backend = NativeDolphin(config)
        backend.load_slot(mission['start_slot'])
        if not start_is_canonical(backend.memory.sample(), config, variation):
            raise ValueError('Replay did not start at the canonical living state.')
        for index in range(len(rows)):
            if time.monotonic() >= deadline or memory_snapshot()['physical_available_bytes'] < 3 * 2**30:
                raise TimeoutError('Preparation time or memory boundary reached.')
            if index:
                backend.advance(tuple(inputs[index - 1]['controls']))
                report['human_replay_samples'] += 1
            t = backend.memory.sample()
            if np.linalg.norm(np.array(t.position) - points[index]) > .1:
                raise ValueError(f'Human replay position diverged at {index}.')
            if not np.array_equal(exact_observation(backend), images[index]):
                raise ValueError(f'Human replay observation diverged at {index}.')
            report['matching_observations'] += 1
            if index in wanted:
                name = f'human-anchor-{wanted[index]:03d}.state'
                path = output / name
                backend.save_state(path)
                report['candidates'].append({
                    'file': name, 'name': name[:-6], 'sha256': digest(path), 'position': list(t.position),
                    'health': t.health, 'scene_frame': t.scene_frame, 'scenario': mission['scenario'],
                    'origin_slot': mission['start_slot'], 'near_star_fixture': False,
                    'route_waypoint': wanted[index], 'region': wanted[index], 'route_sha256': digest(route_path),
                    'human_sample': index, 'accepted': False,
                    'validation': {'passed': False, 'idle_frames': 60, 'repetitions': 2}})
                atomic_json(output / 'preparation.json', report)
            if t.dead:
                raise ValueError('Human replay died before its recorded end.')
            if goal_candidate and index == len(rows) - 1:
                inside = np.linalg.norm(np.array(t.position) - goal) <= calibration['goal_radius_3d']
                report['human_ram_star'] = bool(inside and t.grounded is True)
                break
            if t.complete:
                report['human_ram_star'] = confirmed_star(t, True, index, game_id, mission['scenario'])
                break
        if not report['human_ram_star'] or report['human_replay_samples'] != len(inputs):
            raise ValueError('Exact full-start human completion (or goal-marker arrival) was not reproduced.')
        for candidate in report['candidates']:
            passed = True
            for repetition in range(2):
                backend.load_state(output / candidate['file'])
                restored = backend.memory.sample()
                passed &= (restored.health == candidate['health'] and restored.scene_frame == candidate['scene_frame']
                           and restored.grounded is True and not restored.dead and not restored.complete
                           and np.linalg.norm(np.array(restored.position) - candidate['position']) <= .01)
                for _ in range(30):
                    if time.monotonic() >= deadline:
                        raise TimeoutError('Idle verification deadline.')
                    backend.advance(())
                    t = backend.memory.sample()
                    passed &= not t.dead and not t.complete and t.grounded is True and t.health == candidate['health']
                candidate['validation'][f'probe_{repetition}_end'] = t.to_dict()
            candidate['accepted'] = candidate['validation']['passed'] = bool(passed)
            if not passed:
                candidate['rejection'] = 'Grounded/health-stable idle or exact reset check failed.'
            atomic_json(output / 'preparation.json', report)
        entries = sorted([c for c in report['candidates'] if c['accepted']], key=lambda c: c['route_waypoint'])
        if len(entries) < 3:
            raise ValueError('Insufficient safe anchors along the human route.')
        bank = {'schema': 2, 'game_id': game_id, 'emulator_sha256': digest(config['paths']['dolphin']),
                'route_sha256': digest(route_path), 'source_full_start_ram_star': not goal_candidate,
                'source_goal_candidate_replayed': goal_candidate,
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
    parser.add_argument('route', type=Path)
    parser.add_argument('config', type=Path)
    parser.add_argument('variation')
    parser.add_argument('output', type=Path)
    parser.add_argument('--count', type=int, default=11)
    parser.add_argument('--max-seconds', type=int, default=1500)
    parser.add_argument('--goal-candidate', action='store_true',
                        help='The source is an owner-marked goal recording and the gate is the calibrated end location.')
    args = parser.parse_args()
    print(json.dumps(prepare(args.source, args.route, args.config, args.variation, args.output, args.count,
                             args.max_seconds, args.goal_candidate), indent=2))
