"""Check that three four-frame commands reproduce legacy controls in actual games."""

import argparse
import json
from copy import deepcopy
from pathlib import Path

import numpy as np

from smg_rl.actions import advance_action
from smg_rl.config import load_config
from smg_rl.diagnostics import atomic_json
from smg_rl.native import NativeDolphin
from smg_rl.parallel import runtime_directory
from smg_rl.train import new_run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--remaining-from', type=Path)
    args = parser.parse_args()
    previous = json.loads(args.remaining_from.read_text()) if args.remaining_from else {}
    completed = {(m['game'], m['variation']) for m in previous.get('missions', []) if m['passed']}
    base = load_config('config/regional-smg1-12.yaml')
    base['paths']['runs'] = 'runs/regional-intervention'
    run = new_run(base, 'frame4-equivalence')
    report = {'state': 'running', 'run': str(run), 'missions': [],
              'retained_partial_report': str(args.remaining_from) if args.remaining_from else None,
              'scope': 'Scripted physical-control equivalence only; no learned navigation or replay.',
              'maximum_frames_per_mission': 408}
    # Seventeen legacy decisions x12 frames, replayed using either12 or4-frame commands.
    schedule = [('d',)] * 3 + [('w',)] * 4 + [('jump',)] + [()] * 6 + [('spin',), ('crouch',), ()]
    assert len(schedule) * 12 * 2 == 408
    try:
        for game in ('smg1', 'smg2'):
            config = load_config(f'config/regional-{game}-12.yaml')
            config['paths']['user'] = str(runtime_directory(config, run / game) / 'template')
            backend = NativeDolphin(config)
            try:
                for variation in config['variations']:
                    if (game, variation['name']) in completed:
                        continue
                    samples, images = [], []
                    for mode in ('legacy12', 'matched4'):
                        backend.load_slot(variation['start_slot'])
                        backend._motor_direction = ()
                        execution = deepcopy(config)
                        execution['parallel']['frame_skip'] = 12
                        execution['actions'] = {'profile': 'motor8-v1'}
                        if mode == 'matched4':
                            execution['actions'] = {'profile': 'motor8-timed-v2', 'decision_frames': 4,
                                                     'pulse_frames': 4, 'jump_direction': 'inertia'}
                            execution['parallel']['frame_skip'] = 4
                        backend.config['parallel']['frame_skip'] = execution['parallel']['frame_skip']
                        trace = []
                        frames = 0
                        for names in schedule:
                            commands = [names] if mode == 'legacy12' else (
                                [names, names, ()] if names in (('jump',), ('spin',)) else [names] * 3)
                            for command in commands:
                                advance_action(backend, command, execution)
                                frames += backend.last_frame_count
                            trace.append(backend.memory.sample().to_dict())
                        assert frames == 204
                        samples.append(trace)
                        images.append(backend.image().copy())
                    max_error = max(float(np.linalg.norm(np.subtract(a['position'], b['position'])))
                                    for a, b in zip(*samples))
                    for a, b in zip(*samples):
                        for key in ('scenario', 'scene_frame', 'health', 'grounded', 'dead', 'complete'):
                            assert a[key] == b[key], (game, variation['name'], key, a[key], b[key])
                    assert max_error < .01
                    report['missions'].append({'game': game, 'variation': variation['name'], 'passed': True,
                        'frames': 408, 'maximum_xyz_error': max_error,
                        'final_image_mean_abs_error': float(np.abs(images[0].astype(float) - images[1]).mean()),
                        'twelve_frame_samples': samples[0], 'matched_four_frame_samples': samples[1]})
                    atomic_json(run / 'result.json', report)
            finally:
                backend.close()
        report['state'] = 'completed'
    except BaseException as error:
        report.update(state='failed', error=repr(error))
        raise
    finally:
        atomic_json(run / 'result.json', report)
    print(json.dumps({'state': report['state'], 'run': str(run),
                      'missions': [{k: v for k, v in m.items() if not k.endswith('_samples')} for m in report['missions']]}))


if __name__ == '__main__':
    main()
