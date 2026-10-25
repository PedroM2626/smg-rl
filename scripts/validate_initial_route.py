"""Bounded RAM-grounded SMG1 motor fixtures and a scripted initial-platform exit."""

import csv
import json
from pathlib import Path

from smg_rl.config import load_config
from smg_rl.diagnostics import atomic_json
from smg_rl.native import NativeDolphin
from smg_rl.plotting import plot_trajectory
from smg_rl.policy_video import VideoBackend
from smg_rl.rewards import ProgressReward
from smg_rl.route import RouteProgress


def main():
    config = load_config('config/frontier-smg1.yaml')
    root = Path('runs/scripted-initial-exit-20261025')
    root.mkdir(exist_ok=False)
    config['paths']['user'] = str(root / 'profile' / 'template')
    config['parallel'].update(native_capture='full', frame_skip=2)
    spec = json.loads(Path('config/routes/smg1-initial-exit.json').read_text())
    backend = NativeDolphin(config)
    video = VideoBackend(backend, root / 'scripted-fixtures.mp4',
                         'SCRIPTED CONTROL FIXTURES | not learned behavior or mission completion')
    result = {'scope': 'Scripted motor/contact fixtures, excluded from training and mission-success accounting',
              'physical_frames': 0, 'traces': {}, 'video': str(root / 'scripted-fixtures.mp4')}

    def run_case(name, sequence):
        backend.load_slot(1)
        trace = [backend.memory.sample().to_dict()]
        video.episode += 1
        for names, frames in sequence:
            for _ in range(frames // 2):
                backend.advance(names)
                result['physical_frames'] += 2
                t = backend.memory.sample()
                trace.append(t.to_dict())
                video.telemetry, video.last_action = t, '+'.join(names) or 'idle'
                video.step += 1
                video.frame()
                if t.dead or t.complete:
                    break
            if trace[-1]['dead'] or trace[-1]['complete']:
                break
        result['traces'][name] = trace
        return trace

    try:
        jumping = run_case('jump_and_land', [(('jump',), 12), ((), 72)])
        falling = run_case('fall', [(('w',), 180)])
        safe = run_case('initial_exit', [(('w', 'd'), 54), (('w',), 132)])
        route = RouteProgress(spec)
        from smg_rl.memory import Telemetry

        route.reset(Telemetry(**safe[0]))
        calibration = json.loads(Path(config['paths']['calibration']).read_text())['variations']['sweet_sweet']
        projection = ProgressReward(calibration['start'], calibration['goal'])
        projection.reset(safe[0]['position'])
        rows = []
        for i, row in enumerate(safe):
            t = Telemetry(**row)
            route.update(t)
            _, fraction = projection.update(t.position, t.dead, t.complete)
            rows.append({'step': i, 'step_unit':'Two-frame sample', 'variation': 'sweet_sweet', 'x':t.position[0], 'y':t.position[1],
                         'z':t.position[2], 'progress':fraction, 'best_progress':projection.best,
                         'route_prefix_progress':route.best, 'route_prefix_exited':route.exited,
                         'grounded': t.grounded, 'health':t.health, 'outcome':'scripted_fixture'})
        result['checks'] = {
            'initial_grounded': jumping[0]['grounded'] is True,
            'jump_has_airborne_samples': any(r['grounded'] is False and r['position'][1] > 100 for r in jumping),
            'landed_again': jumping[-1]['grounded'] is True and not jumping[-1]['dead'],
            'fall_has_airborne_samples': any(r['grounded'] is False and r['position'][1] < -200 for r in falling),
            'scripted_safe_exit': route.exited,
            'safe_exit_health_preserved': all(r['health'] == 3 and not r['dead'] for r in safe),
            'no_mission_completion': not any(r['complete'] for trace in result['traces'].values() for r in trace)}
        with (root / 'trajectory.csv').open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=rows[0])
            writer.writeheader()
            writer.writerows(rows)
        plot_trajectory(root / 'trajectory.csv')
    finally:
        video.close()
    result['state'] = 'passed' if all(result['checks'].values()) else 'failed'
    atomic_json(root / 'result.json', result)
    print(json.dumps({'run':str(root), 'state':result['state'], 'checks':result['checks']}))
    if result['state'] != 'passed':
        raise RuntimeError('Control/contact validation failed.')


if __name__ == '__main__':
    main()
