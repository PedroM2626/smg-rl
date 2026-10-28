"""Offline counterfactual over retained, full-start expanding-route episode CSVs."""

import argparse
import csv
import json
import statistics
from pathlib import Path

from smg_rl.diagnostics import atomic_json


def audit(state_path):
    state = json.loads(Path(state_path).read_text(encoding='utf-8'))
    episodes, partial, violations = [], 0, []
    for block in state['blocks']:
        for path in Path(block['run']).glob('worker-*/*/episode-*.csv'):
            with path.open(newline='', encoding='utf-8') as stream:
                rows = list(csv.DictReader(stream))
            if not rows or rows[0]['start_kind'] != 'full_start':
                continue
            if rows[-1]['outcome'] not in ('death', 'timeout', 'success'):
                partial += 1
                continue
            shaping = 0.
            other = 0.
            for previous, row in zip(rows, rows[1:]):
                amount = float(row['regional_shaping_reward'])
                shaping += amount
                other += float(row['reward']) - amount
                still = all(abs(float(row[k]) - float(previous[k])) < 1e-5 for k in ('x', 'y', 'z'))
                if amount > 1e-8 and (still or row['grounded'] != 'True' or row['dead'] == 'True'):
                    violations.append({'trajectory': str(path), 'step': row['step'], 'stationary': still,
                                       'grounded': row['grounded'], 'dead': row['dead'], 'reward': amount})
            episodes.append({'outcome': rows[-1]['outcome'], 'advance': shaping,
                             'observed_return': shaping + other, 'candidate_return': 10 * shaping + other})
    by_outcome = {}
    for outcome in ('death', 'timeout', 'success'):
        subset = [entry for entry in episodes if entry['outcome'] == outcome]
        if subset:
            by_outcome[outcome] = {'episodes': len(subset),
                'median_advance': statistics.median(e['advance'] for e in subset),
                'max_advance': max(e['advance'] for e in subset),
                'mean_observed_return': statistics.mean(e['observed_return'] for e in subset),
                'mean_counterfactual_return': statistics.mean(e['candidate_return'] for e in subset)}
    return {'completed': len(episodes), 'partial_excluded': partial, 'by_outcome': by_outcome,
            'positive_reward_on_identical_xyz_airborne_or_dead': violations,
            'counterfactual': 'Retained decisions only; tenfold route credit assumes same trajectory and no saturation. It is not training or causal evidence.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--state', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    result = audit(args.state)
    if not atomic_json(args.output, result):
        raise OSError('Cannot save reward audit.')
    print(json.dumps(result, indent=2))
