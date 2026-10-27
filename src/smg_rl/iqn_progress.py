"""Read-only canonical IQN progress toward the calibrated star, across the full coordinate span."""

import csv
import json
from datetime import datetime, timezone
from pathlib import Path

from smg_rl.config import load_config
from smg_rl.diagnostics import atomic_json
from smg_rl.dqn_trial import locate_run, read
from smg_rl.genetic import GroundedAdvance
from smg_rl.profiles import PROFILES


def episode_projection(path, bounds, max_step):
    with Path(path).open(newline='', encoding='utf-8') as stream:
        rows = list(csv.DictReader(stream))
    if not rows or rows[0].get('start_kind') != 'full_start':
        return {'excluded': 'assisted_or_missing_start'}
    star = rows[-1]['complete'] == 'True'
    if star and int(rows[-1]['scene_nerve'], 16) not in PROFILES['RMGE01'].complete:
        raise ValueError('An IQN star claim lacks native completion RAM.')
    if any(row.get('grounded') not in ('True', 'False') for row in rows):
        return {'excluded': 'missing_grounded_signal', 'ram_star': star}
    tracker = GroundedAdvance(bounds['start'], bounds['goal'], max_step)
    best = None
    for row in rows[1:]:
        before = tracker.best
        tracker.update({'position': [float(row[k]) for k in ('x', 'y', 'z')],
                        'grounded': row['grounded'] == 'True', 'dead': row['dead'] == 'True'})
        if tracker.best > before:
            best = row
    if tracker.invalid:
        return {'excluded': 'nonfinite_or_discontinuous_position', 'ram_star': star}
    point = best or rows[0]
    return {'trajectory': str(path), 'geometric_projection_percent': 100 * tracker.best / tracker.length,
            'grounded_fitness_world_units': tracker.best, 'step': int(point['step']),
            'position': [float(point[k]) for k in ('x', 'y', 'z')], 'outcome': rows[-1]['outcome'],
            'saved_decisions': len(rows) - 1, 'ram_star': star, 'start_kind': 'full_start',
            'partial_episode': rows[-1]['outcome'] in ('running', 'interrupted'),
            'mapped_opening_fraction': float(rows[-1].get('route_prefix_progress') or 0)}


def scan_runs(runs, bounds, max_step):
    result = {'runs': len(runs), 'canonical_saved_episodes': 0, 'saved_decisions': 0,
              'native_full_start_stars': 0, 'excluded_episodes': {}, 'best': None}
    seen = set()
    for run in runs:
        for index in run.glob('worker-*/episodes.jsonl'):
            for line in index.read_text().splitlines():
                try:
                    episode = json.loads(line)
                except json.JSONDecodeError:
                    continue  # A current writer may not have completed its last line yet.
                if episode.get('variation') != 'sweet_sweet':
                    continue
                path = index.parent / 'sweet_sweet' / f"episode-{episode['episode']:06d}.csv"
                if path in seen:
                    continue
                seen.add(path)
                value = episode_projection(path, bounds, max_step)
                result['native_full_start_stars'] += int(value.get('ram_star', False))
                if 'excluded' in value:
                    key = value['excluded']
                    result['excluded_episodes'][key] = result['excluded_episodes'].get(key, 0) + 1
                    continue
                result['canonical_saved_episodes'] += 1
                result['saved_decisions'] += value['saved_decisions']
                if result['best'] is None or value['grounded_fitness_world_units'] > result['best']['grounded_fitness_world_units']:
                    result['best'] = value
    return result


def report_iqn_progress(state_path, output):
    state = read(Path(state_path))
    branch = state['games']['smg1']
    config = load_config(branch['config'])
    if config['game']['id'] != 'RMGE01' or config['dqn']['algorithm'] != 'iqn':
        raise ValueError('This report requires the SMG1 IQN experiment ledger.')
    bounds = read(Path(config['paths']['calibration']))['variations']['sweet_sweet']
    roots = ['runs/model-free/iqn/smg1', 'runs/model-free/simple-impala-iqn/smg1', 'runs/model-free/smg1-six-two-hour']
    baseline = []
    for root in roots:
        for run in (Path(config['root']) / root).glob('dqn-*'):
            if not (run / 'controller.json').exists():
                continue
            controller = read(run / 'controller.json')
            if (controller.get('algorithm') == 'iqn' and controller.get('requested_additional')
                    and controller.get('source_count', float('inf')) < branch['source_real']):
                baseline.append(run)
    current = []
    for job in branch['jobs']:
        if not Path(job['stdout']).exists():
            continue  # Pending dispatch has not created its output log yet.
        run = locate_run(job)
        if run and run not in current:
            current.append(run)
    report = {'recorded_utc': datetime.now(timezone.utc).isoformat(), 'state': str(state_path),
              'source_real_decisions': branch['source_real'], 'training_phase': state['phase'],
              'metric': 'Maximum alive-grounded X/Z projection from canonical start toward calibrated near-star goal; not native mission percentage or full-route arc length.',
              'baseline_before_phase': scan_runs(baseline, bounds, config['environment']['max_step_distance']),
              'current_phase': scan_runs(current, bounds, config['environment']['max_step_distance']),
              'coverage': 'Saved canonical episode CSVs, including saved collector cuts. Active unflushed episodes are absent while training runs. Not a frozen-policy evaluation; exploration and changing policies are included.'}
    if not atomic_json(Path(output), report):
        raise OSError('Could not persist the read-only IQN progress report.')
    return report
