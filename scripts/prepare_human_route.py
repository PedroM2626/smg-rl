"""Prepare a separate signed route candidate from a replayed human completion."""

import argparse
import csv
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import yaml

from smg_rl.checkpoints import digest
from smg_rl.route import RouteProgress


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('demonstration', type=Path)
    parser.add_argument('replay_result', type=Path)
    parser.add_argument('--route', required=True, type=Path)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--report', required=True, type=Path)
    args = parser.parse_args()
    source = args.demonstration.resolve()
    manifest = json.loads((source / 'manifest.json').read_text(encoding='utf-8'))
    replay = json.loads(args.replay_result.read_text(encoding='utf-8'))
    game_id = manifest['game_id']
    base = 'config/smg1-six-two-hour.yaml' if game_id == 'RMGE01' else 'config/impala-iqn-smg2.yaml'
    profile = yaml.safe_load(Path(base).read_text(encoding='utf-8'))
    matches = [v for v in profile['variations'] if v['scenario'] == manifest['scenario']
               and v['start_slot'] == manifest['start_slot']]
    if (manifest['state'] != 'completed' or game_id not in ('RMGE01', 'SB4E01')
            or len(matches) != 1
            or not manifest['full_start_ram_star'] or replay['state'] != 'star'
            or Path(replay['source']).resolve() != source or not replay['ram_star']
            or replay['real_replay_samples'] != manifest['samples']
            or replay['first_position_divergence'] is not None
            or replay['first_observation_divergence'] is not None
            or not replay['source_savestate_unchanged']):
        raise ValueError('A matching, exactly replayed canonical human star is required.')
    for name, value in manifest['files_sha256'].items():
        if Path(name).name != name or digest(source / name) != value:
            raise ValueError(f'Invalid source artifact: {name}')
    with (source / 'trajectory.csv').open(encoding='utf-8', newline='') as stream:
        rows = list(csv.DictReader(stream))
    if (len(rows) != manifest['samples'] + 1 or rows[0]['grounded'] != 'True'
            or rows[-1]['complete'] != 'True' or any(row['dead'] == 'True' for row in rows)):
        raise ValueError('Invalid full-start trajectory.')
    positions = np.asarray([[float(row[k]) for k in ('x', 'y', 'z')] for row in rows])
    if not np.isfinite(positions).all():
        raise ValueError('Non-finite trajectory position.')
    grounded = [row['grounded'] == 'True' for row in rows]
    indices = [0]
    for i in range(1, len(rows) - 1):
        if grounded[i] and np.linalg.norm((positions[i] - positions[indices[-1]])[[0, 2]]) >= 400:
            indices.append(i)
    mission = matches[0]
    spec = {'schema': 4, 'game_id': game_id, 'variation': mission['name'],
            'name': 'Human full-start grounded waypoint candidate',
            'scope': 'Research candidate, derived from one replayed completion; route end is not a RAM star.',
            'waypoints': positions[indices].tolist(), 'waypoint_radius': 100,
            'corridor_radius': 100, 'height_tolerance': 500, 'exit_confirmations': 1,
            'max_sample_distance': 1000, 'requires_grounded': True, 'requires_full_health': False,
            'source_inputs_sha256': manifest['files_sha256']['inputs.jsonl'],
            'source_state_sha256': manifest['source_state_sha256']}
    route = RouteProgress(spec)

    def telemetry(i):
        return SimpleNamespace(position=positions[i], grounded=grounded[i],
                               health=int(rows[i]['health']) if rows[i]['health'] not in ('', 'None') else None,
                               dead=rows[i]['dead'] == 'True', complete=rows[i]['complete'] == 'True')

    phases = []
    for offset in range(6):
        route.reset(telemetry(0))
        for i in range(1 + offset, len(rows) - 1, 6):
            route.update(telemetry(i))
        phases.append(route.best)
    if min(phases) < .97:
        raise ValueError('The route cannot follow the demonstrated path at motor8 observation intervals.')
    if any(p.exists() for p in (args.route, args.config, args.report)):
        raise FileExistsError('Output artifacts are immutable; choose unused paths.')
    args.route.parent.mkdir(parents=True, exist_ok=True)
    # Route bytes are signed and preserved verbatim across platforms.
    args.route.write_bytes((json.dumps(spec, indent=2) + '\n').encode('utf-8'))
    profile['variations'] = [mission]  # Each game/mission gets its own signed route and replay.
    profile['paths']['route'] = args.route.as_posix()
    prefix = f'human-route-{game_id.lower()}-{mission["name"]}-candidate'
    profile['paths']['runs'] = f'runs/model-free/{prefix}'
    profile['paths']['replay'] = f'runs/model-free/{prefix}/replay'
    profile['paths']['user'] = f'.local/{prefix}/template'
    profile['curriculum']['enabled'] = False
    profile['regional_reward']['spec_sha256'] = digest(args.route)
    profile['regional_reward']['normalization_distance'] = float(route.total_length)
    # Keep the original bounded objective: complete route credit is at most 1,
    # less than the magnitude of the -2 death/timeout penalties.
    profile['regional_reward']['scale'] = 1.0
    profile['rewards']['revision'] = 'human-full-route-swept-v1'
    # This is a distinct experimental objective, not an exact checkpoint resume.
    profile['training']['enabled'] = True
    args.config.parent.mkdir(parents=True, exist_ok=True)
    args.config.write_text(yaml.safe_dump(profile, sort_keys=False), encoding='utf-8')
    report = {'state': 'offline_candidate', 'source_input_sha256': spec['source_inputs_sha256'],
              'route_sha256': digest(args.route), 'config_sha256': digest(args.config),
              'waypoints': len(indices), 'waypoint_samples': indices,
              'horizontal_route_length': float(route.total_length),
              'fraction_by_twelve_frame_phase': phases,
              'training_enabled': True, 'native_completion': 'Separate RAM signal only.',
              'limitations': ['Same human trajectory and same initial state; not unseen validation.',
                              'Reward exploitation and performance in learned policies remain untested.',
                              'No control or image conversion; no training steps.']}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
