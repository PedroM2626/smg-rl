"""Bounded contact/reward fixtures and reachable intermediate-state candidates."""

import argparse
import json
import os

import cv2
import psutil

from smg_rl.actions import advance_action
from smg_rl.checkpoints import digest
from smg_rl.config import load_config
from smg_rl.diagnostics import atomic_json
from smg_rl.native import NativeDolphin
from smg_rl.parallel import runtime_directory
from smg_rl.train import new_run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--game', choices=['smg1', 'smg2'], required=True)
    args = parser.parse_args()
    config = load_config(f'config/motor8-{args.game}.yaml')
    config['actions'] = {'profile': 'motor8-timed-v2', 'decision_frames': 4,
                         'pulse_frames': 4, 'jump_direction': 'carry'}
    config['parallel'].update(frame_skip=4, native_capture='full')
    config['replay']['enabled'] = False
    config['paths']['runs'] = 'runs/regional-intervention'
    run = new_run(config, f'prepare-{args.game}')
    config['paths']['user'] = str(runtime_directory(config, run) / 'template')
    report = {'state': 'running', 'game': args.game, 'run': str(run),
              'emulator_sha256': digest(config['paths']['dolphin']), 'missions': [],
              'maximum_frames_per_mission': 6000,
              'scope': 'Scripted geometry/contact/curriculum fixtures only. No learner, training replay or learned success.'}
    atomic_json(run / 'controller.json', {'pid': os.getpid(), 'created_time': psutil.Process().create_time(),
                                         'maximum_frames_per_mission': 6000, 'training_allowed': False})
    print(json.dumps({'run': str(run)}), flush=True)
    backend = NativeDolphin(config)
    try:
        for variation in config['variations']:
            mission = {'variation': variation['name'], 'scenario': variation['scenario'],
                       'origin_slot': variation['start_slot'], 'physical_frames': 0,
                       'fixtures': {}, 'candidates': []}
            report['missions'].append(mission)
            traces = {}

            def reset():
                backend.load_slot(variation['start_slot'])
                backend._motor_direction = ()
                return backend.memory.sample()

            def execute(name, schedule):
                trace = [{'telemetry': backend.memory.sample().to_dict(), 'logical': [], 'frames': 0}]
                for names, frames in schedule:
                    for _ in range(frames // 4):
                        if mission['physical_frames'] + 4 > 6000:
                            raise RuntimeError('Finite preparation frame budget exhausted.')
                        advance_action(backend, tuple(names), config)
                        mission['physical_frames'] += 4
                        sample = backend.memory.sample()
                        trace.append({'telemetry': sample.to_dict(), 'logical': names, 'frames': 4})
                        if sample.dead or sample.complete:
                            break
                    if trace[-1]['telemetry']['dead'] or trace[-1]['telemetry']['complete']:
                        break
                traces[name] = trace
                rgba = backend.bridge.image()
                cv2.imwrite(str(run / f"{variation['name']}-{name}.png"), cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA))
                return trace

            initial = reset()
            jump = execute('jump-land', [(['jump'], 12), ([], 84)])
            reset()
            fall = execute('fall', [(['d'], 200)])
            mission['fixtures']['contact'] = {
                'initial_grounded': initial.grounded is True,
                'airborne_jump': any(t['telemetry']['grounded'] is False and
                    t['telemetry']['position'][1] > initial.position[1] + 20 for t in jump),
                'landed': jump[-1]['telemetry']['grounded'] is True and not jump[-1]['telemetry']['dead'],
                'airborne_fall': any(t['telemetry']['grounded'] is False for t in fall),
                'ram_death': fall[-1]['telemetry']['dead']}
            if not all(mission['fixtures']['contact'].values()):
                raise RuntimeError('Game-specific contact/death fixture failed.')
            if args.game == 'smg1':
                schedules = {
                    'right-approach': [(['d'], 36), ([], 24)],
                    'first-pink': [(['d'], 36), (['w'], 192), ([], 12)],
                    'first-chocolate': [(['d'], 36), (['w'], 192), (['d'], 24), (['w'], 96), ([], 24)],
                    'beyond-chocolate': [(['d'], 36), (['w'], 192), (['d'], 24), (['w'], 80),
                                         (['jump'], 24), (['w'], 72), ([], 12)]}
            else:
                schedules = {
                    'platform-approach': [(['w'], 48), ([], 24)],
                    'first-bridge': [(['w'], 120), ([], 12)],
                    'obstacle-approach': [(['w'], 160), ([], 12)],
                    'beyond-obstacle': [(['w'], 172), (['jump'], 24), (['w'], 64), ([], 12)]}
            for name, schedule in schedules.items():
                reset()
                execute(name, schedule)
                before = backend.memory.sample()
                entry = {'name': name, 'scenario': variation['scenario'], 'origin_slot': variation['start_slot'],
                         'schedule': schedule, 'position': list(before.position), 'health': before.health,
                         'scene_frame': before.scene_frame, 'near_star_fixture': False,
                         'accepted': False, 'validation': {'passed': False, 'idle_frames': 60}}
                mission['candidates'].append(entry)
                if (before.dead or before.complete or before.grounded is not True
                        or (initial.health is not None and before.health != initial.health)):
                    entry['rejection'] = 'Not an alive grounded undamaged candidate.'
                    continue
                path = run / f"{variation['name']}-{name}.state"
                backend.save_state(path)
                entry.update(file=str(path), sha256=digest(path))
                passed = True
                for repetition in range(2):
                    backend.load_state(path)
                    backend._motor_direction = ()
                    probe = execute(f'{name}-idle{repetition}', [([], 60)])
                    passed &= all(not t['telemetry']['dead'] and not t['telemetry']['complete']
                                  and t['telemetry']['grounded'] is True
                                  and t['telemetry']['health'] == before.health for t in probe)
                backend.load_state(path)
                restored = backend.memory.sample()
                if (restored.position != before.position or restored.scene_frame != before.scene_frame
                        or restored.health != before.health or restored.grounded is not True):
                    raise RuntimeError('Candidate probe did not restore its exact context.')
                entry['accepted'] = entry['validation']['passed'] = bool(passed)
                if not passed:
                    entry['rejection'] = 'Failed repeated60-frame grounded idle safety probe.'
                print(json.dumps({'variation': variation['name'], 'candidate': name,
                                  'accepted': entry['accepted'], 'position': entry['position']}), flush=True)
                atomic_json(run / 'result.json', report)
            # Original near-star slots remain detector fixtures, never accepted starts.
            backend.load_slot(variation['goal_slot'])
            backend._motor_direction = ()
            star = execute('near-star-detector-only', [(['jump'], 48), ([], 48)])
            mission['fixtures']['near_star_detector'] = {'complete': star[-1]['telemetry']['complete'],
                'source_slot': variation['goal_slot'], 'training_or_full_start_success': False}
            atomic_json(run / f"{variation['name']}-traces.json", traces)
            atomic_json(run / 'result.json', report)
        report['state'] = 'completed'
    except BaseException as error:
        report.update(state='failed', error=repr(error))
        raise
    finally:
        backend.close()
        atomic_json(run / 'result.json', report)
    print(json.dumps({'state': report['state'], 'run': str(run),
                      'accepted': {m['variation']: sum(c['accepted'] for c in m['candidates']) for m in report['missions']}}), flush=True)


if __name__ == '__main__':
    main()
