"""Re-score retained frozen-policy evaluations against one fixed human route."""

import csv
import json
from pathlib import Path

from smg_rl.checkpoints import digest
from smg_rl.human_route_audit import load_spec, score_episode
from smg_rl.profiles import PROFILES

ROUTE = Path('config/routes/smg1-human-landing-v2.json')
STUDIES = {'reward_scale': Path('.local/reward-scale-ab-20261028/state.json'),
           'chocolate_demo': Path('.local/chocolate-demo-ab-20261028/state.json')}
OUTPUT = Path('docs/assets/performance/human-route-historical-frozen-20261028.json')


def main():
    if OUTPUT.exists():
        raise FileExistsError('Historical audit evidence must not be overwritten.')
    route = load_spec(ROUTE)
    output = {'route_sha256': digest(ROUTE), 'studies': {}, 'episodes': [], 'errors': [],
              'scope': 'Post-hoc re-scoring of old frozen full-start evaluations; no new training or testing.',
              'limitation': 'Reused saved trajectories and the same Wii state; waypoint fractions are not native completion.'}
    for label, path in STUDIES.items():
        try:
            state = json.loads(path.read_text(encoding='utf-8'))
            cases = state.get('runs', state.get('cases', []))
            if len(cases) != 4:
                raise ValueError('Expected four reviewed cases.')
            output['studies'][label] = {'source_sha256': digest(path), 'phase': state['phase']}
            for case in cases:
                for ev in case['evaluations']:
                    if ev['phase'] != 'reviewed':
                        raise ValueError('An evaluation is not reviewed.')
                    run = Path(ev['run'])
                    result = json.loads((run / 'recording.json').read_text(encoding='utf-8'))
                    if result['state'] != 'completed' or result['agent_sha256'] != ev['result']['agent_sha256']:
                        raise ValueError('Evaluation result identity or status mismatch.')
                    for index, episode in enumerate(result['episodes'], 1):
                        trajectory = run / 'worker-00' / 'sweet_sweet' / f'episode-{index:06d}.csv'
                        score = score_episode(trajectory, route)
                        with trajectory.open(encoding='utf-8', newline='') as stream:
                            rows = list(csv.DictReader(stream))
                        if score['native_star'] != bool(episode['full_start_ram_star']):
                            raise ValueError('Native star flags disagree.')
                        if score['native_star'] and int(rows[-1]['scene_nerve'], 16) not in PROFILES['RMGE01'].complete:
                            raise ValueError('Star lacks the expected RAM terminal.')
                        output['episodes'].append({'study': label, 'arm': case['arm'],
                            'training_seed': case['seed'], 'mode': ev['mode'], 'action_seed': episode['seed'],
                            'score': score, 'outcome': episode['outcome'],
                            'hp_lost': sum(int(row['health_lost']) for row in rows[1:]),
                            'goal_axis_projection_percent': episode['maximum_geometric_projection_percent'],
                            'trajectory': str(trajectory), 'trajectory_sha256': digest(trajectory)})
        except (OSError, ValueError, KeyError) as exc:
            output['errors'].append({'study': label, 'error': repr(exc)})
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(output, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'output': str(OUTPUT), 'episodes': len(output['episodes']),
                      'errors': output['errors']}), flush=True)


if __name__ == '__main__':
    main()
