"""Counterfactual reward audit on retained real SMG1 episodes; never rewrites replay."""

import argparse
import csv
import json
from pathlib import Path

from smg_rl.checkpoints import digest
from smg_rl.config import load_config
from smg_rl.diagnostics import atomic_json
from smg_rl.memory import Telemetry
from smg_rl.rewards import RegionalPotential
from smg_rl.route import RouteProgress


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', default='docs/assets/performance/motor8-reward-audit-20261025.json')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    config = load_config('config/regional-smg1-12.yaml')
    source = json.loads(Path(args.source).read_text())
    spec = json.loads(Path(config['paths']['route']).read_text())
    settings = config['regional_reward']
    results = []
    for episode in source['episodes']:
        if episode['outcome'] not in ('death', 'timeout'):
            continue  # Collector cuts are not terminal failures.
        path = Path(episode['csv'])
        if digest(path) != episode['sha256']:
            raise ValueError('Retained source episode changed.')
        rows = list(csv.DictReader(path.open(newline='')))

        def sample(row):
            if row['grounded'] not in ('True', 'False'):
                raise ValueError('Cannot invent absent contact labels.')
            return Telemetry(tuple(float(row[k]) for k in ('x', 'y', 'z')), 1, 'BeltConveyerExGalaxy',
                int(row['actor'], 16), int(row['actor_nerve'], 16), int(row['scene_nerve'], 16),
                int(row['scene_frame']), row['dead'] == 'True', row['complete'] == 'True',
                health=int(row['health']) if row['health'] else None, grounded=row['grounded'] == 'True')

        route = RouteProgress(spec)
        route.reset(sample(rows[0]))
        potential = RegionalPotential(settings['scale'], settings['gamma_per_frame'])
        potential.reset(0.)
        discount, shaping, total = 1., 0., 0.
        for index, row in enumerate(rows[1:], 1):
            telemetry = sample(row)
            route.update(telemetry)
            terminal = index == len(rows) - 1
            value = min(1., route.best * route.total_length / settings['normalization_distance'])
            amount = potential.update(value, 12, terminal=terminal)
            shaping += discount * amount
            base = config['rewards']['step'] + float(row['damage_penalty'])
            if terminal:
                base += config['rewards']['death' if episode['outcome'] == 'death' else 'timeout']
            total += discount * (base + amount)
            discount *= settings['gamma_per_frame'] ** 12
        assert abs(shaping) < 1e-10 and total < 0
        results.append({'csv': str(path), 'sha256': digest(path), 'outcome': episode['outcome'],
                        'old_discounted_return': episode['observed_returns']['discounted'],
                        'new_discounted_return': total, 'discounted_shaping': shaping})
    output = {'scope': 'Offline counterfactual on retained real episodes, not unseen evaluation or learning.',
              'source': args.source, 'source_sha256': digest(args.source), 'episodes': results,
              'terminal_failures': len(results),
              'old_positive_failures': sum(e['old_discounted_return'] > 0 for e in results),
              'new_positive_failures': sum(e['new_discounted_return'] > 0 for e in results),
              'route_sha256': settings['spec_sha256']}
    atomic_json(args.output, output)
    print(json.dumps({k: v for k, v in output.items() if k != 'episodes'}))


if __name__ == '__main__':
    main()
