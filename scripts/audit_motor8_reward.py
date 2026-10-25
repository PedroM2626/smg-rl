"""Read-only reward-label audit; counterfactual traces are not trained policies."""

import argparse
import csv
import json
from pathlib import Path
from types import SimpleNamespace

from smg_rl.checkpoints import digest
from smg_rl.config import load_config
from smg_rl.diagnostics import atomic_json
from smg_rl.route import RouteProgress


def telemetry(row):
    return SimpleNamespace(position=tuple(float(row[k]) for k in ('x', 'y', 'z')),
                           grounded=row['grounded'] == 'True',
                           health=int(row['health']) if row['health'] else None,
                           dead=row['dead'] == 'True', complete=row['complete'] == 'True')


def returns(rewards, gamma):
    return {'undiscounted': sum(rewards),
            'discounted': sum(gamma ** i * reward for i, reward in enumerate(rewards))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError('Retain previous reward audits.')
    config = load_config('config/motor8-smg1.yaml')
    gamma = config['world_model']['gamma']
    paths = [Path('config/routes/smg1-initial-exit.json'),
             Path('config/routes/smg1-initial-exit-v2-candidate.json')]
    specs = [json.loads(p.read_text()) for p in paths]
    state = json.loads(Path('.local/motor8-state.json').read_text())
    job = next(j for j in state['jobs'] if j['game'] == 'smg1')
    result = {'scope': 'Read-only actual bootstrap labels, plus offline counterfactual route replay. '
              'No optimizer, emulator or training counters changed; all policies were fixed priors. '
              'Positive returns alone do not establish learned exploitation.',
              'gamma': gamma, 'route_specs': [{'file': str(p), 'sha256': digest(p)} for p in paths],
              'episodes': [], 'fixture': {}}
    for run_name in job['runs']:
        run = Path(run_name)
        if not run.name.startswith('world-model-2'):
            continue
        status = json.loads((run / 'status.json').read_text())
        if status['state'] != 'completed' or status['real_decisions'] > 4096:
            continue
        if status['actor_updates'] != 0:
            raise ValueError('This audit is restricted to fixed-prior bootstrap collection.')
        for path in sorted(run.glob('worker-*/*/episode-*.csv')):
            with path.open(newline='', encoding='utf-8') as stream:
                rows = list(csv.DictReader(stream))
            if rows[0]['start_kind'] != 'full_start' or rows[0]['action'] != 'reset':
                raise ValueError('A full-start reset is required.')
            routes = [RouteProgress(spec) for spec in specs]
            for route in routes:
                route.reset(telemetry(rows[0]))
            observed, original_route, candidate_route = [], [], []
            for row in rows[1:]:
                observed.append(float(row['reward']))
                amounts = [route.update(telemetry(row)) * config['route_reward']['weight']
                           for route in routes]
                recorded = float(row['route_reward'])
                if abs(amounts[0] - recorded) > 1e-8:
                    raise ValueError(f'Historical route labels do not reproduce: {path}')
                original_route.append(recorded)
                candidate_route.append(amounts[1])
            counterfactual = [r - old + new for r, old, new in
                              zip(observed, original_route, candidate_route)]
            result['episodes'].append({
                'csv': str(path), 'sha256': digest(path), 'decisions': len(observed),
                'outcome': rows[-1]['outcome'], 'complete': rows[-1]['complete'] == 'True',
                'prefix_exit': rows[-1]['route_prefix_exited'] == 'True',
                'intrinsic_total': sum(float(r['intrinsic_reward']) for r in rows),
                'observed_returns': returns(observed, gamma),
                'route_payout': sum(original_route),
                'candidate_route_payout': sum(candidate_route),
                'candidate_returns_same_weight': returns(counterfactual, gamma)})
    deaths = [e for e in result['episodes'] if e['outcome'] == 'death']
    result['summary'] = {
        'real_decisions': sum(e['decisions'] for e in result['episodes']),
        'episodes': len(result['episodes']), 'deaths': len(deaths),
        'positive_death_returns': {
            kind: sum(e['observed_returns'][kind] > 0 for e in deaths)
            for kind in ('undiscounted', 'discounted')},
        'candidate_positive_death_returns_same_weight': {
            kind: sum(e['candidate_returns_same_weight'][kind] > 0 for e in deaths)
            for kind in ('undiscounted', 'discounted')}}
    if result['summary']['real_decisions'] != 4096:
        raise ValueError('Expected the exact retained SMG1 bootstrap, with no duplicate runs.')
    for spec in specs:
        route = RouteProgress(spec)
        initial = SimpleNamespace(position=spec['waypoints'][0], grounded=True, health=3,
                                  dead=False, complete=False)
        stationary = SimpleNamespace(**{**vars(initial), 'position': [146, 45, -295]})
        route.reset(initial)
        first, second = route.update(stationary), route.update(stationary)
        result['fixture'][str(spec['schema'])] = {
            'position': stationary.position, 'first_delta': first, 'stationary_delta': second,
            'stationary_reward_at_weight_5': 5 * second,
            'scope': 'Synthetic regression fixture, not game play or learned progress.'}
    atomic_json(output, result)
    print(json.dumps({'summary': result['summary'], 'fixture': result['fixture']}, indent=2))


if __name__ == '__main__':
    main()
