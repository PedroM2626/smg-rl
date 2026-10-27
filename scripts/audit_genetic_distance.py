"""Audit a completed route-free genetic run against its retained trajectories."""

import argparse
import csv
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch

from smg_rl.checkpoints import digest
from smg_rl.goal_distance import GroundedGoalDistance
from smg_rl.profiles import PROFILES


def audit_trial(record, bounds, max_step):
    path = Path(record['trajectory']) if record.get('trajectory') else None
    if path is None or not path.exists():
        if record['fitness_eligible']:
            raise ValueError('Completed trial is missing its trajectory.')
        return None
    with path.open(newline='', encoding='utf-8') as stream:
        rows = list(csv.DictReader(stream))
    if not rows or rows[0]['start_kind'] != 'full_start' or len(rows) - 1 != record['decisions']:
        raise ValueError('Trajectory start or decision count mismatch.')
    score = GroundedGoalDistance(bounds['start'], bounds['goal'],
                                 [float(rows[0][k]) for k in ('x', 'y', 'z')],
                                 max_step, dimensions=(0, 1, 2))
    initial = score.best_distance
    progression = [0.0]
    for row in rows[1:]:
        score.update([float(row[k]) for k in ('x', 'y', 'z')], row['grounded'] == 'True',
                     row['dead'] == 'True', row['complete'] == 'True')
        progression.append(100 * (initial - score.best_distance) / score.normalization)
    if score.invalid or not math.isclose(initial - score.best_distance, record['fitness_world_units'], abs_tol=1e-5):
        raise ValueError('Trajectory does not reproduce distance fitness.')
    star = rows[-1]['complete'] == 'True'
    if star and int(rows[-1]['scene_nerve'], 16) not in PROFILES['RMGE01'].complete:
        raise ValueError('Star lacks game-specific RAM completion evidence.')
    if star != record['ram_star']:
        raise ValueError('Star flag and trajectory differ.')
    return rows, progression


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--figure', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.figure.exists():
        parser.error('Refusing to overwrite existing evidence.')
    summary = json.loads((args.run / 'summary.json').read_text(encoding='utf-8'))
    config = json.loads((args.run / 'configuration.json').read_text(encoding='utf-8'))
    if (summary['state'] != 'completed' or summary['error'] or not summary['source_unchanged']
            or summary['fitness_mode'] != 'goal-distance-xyz-v1'
            or config['regional_reward']['enabled'] or config['curriculum']['enabled']
            or 'route' in config['paths']):
        raise ValueError('Run is incomplete or does not have isolated route-free distance fitness.')
    source = Path(summary['source'])
    if any(digest(source / name) != sha for name, sha in summary['source_hashes'].items()):
        raise ValueError('Frozen source checkpoint changed.')
    if digest(args.run / 'champion.pt') != summary['champion_sha256']:
        raise ValueError('Champion digest mismatch.')
    champion = torch.load(args.run / 'champion.pt', map_location='cpu', weights_only=True)
    source_agent = torch.load(source / 'agent.pt', map_location='cpu', weights_only=True)
    if not all(torch.equal(t, champion['source_online'][key]) for key, t in source_agent['online'].items()):
        raise ValueError('Frozen source network tensors changed.')
    ancestry = summary['initial_head_ancestry']
    if digest(ancestry['path']) != ancestry['sha256'] or ancestry['reused_fitness_trials'] != 0:
        raise ValueError('Initial champion integrity or fitness isolation failed.')
    records = [json.loads(line) for line in (args.run / 'trials.jsonl').read_text().splitlines()]
    if (sum(r['decisions'] for r in records) != summary['real_decisions']
            or sum(r['fitness_eligible'] for r in records) != summary['completed_trials']
            or sum(not r['fitness_eligible'] for r in records) != summary['partial_trials']
            or sum(bool(r['ram_star']) for r in records if r['fitness_eligible']) != summary['full_start_stars']):
        raise ValueError('Ledger and summary counts do not reconcile.')
    bounds = json.loads(Path(config['paths']['calibration']).read_text(encoding='utf-8'))['variations']['sweet_sweet']
    reviewed = 0
    missing_partial = 0
    best_plot = None
    for record in records:
        reviewed_rows = audit_trial(record, bounds, config['environment']['max_step_distance'])
        if reviewed_rows is None:
            missing_partial += 1
            continue
        reviewed += 1
        if (record['genome_sha256'] == summary['champion']['genome_sha256']
                and record['trajectory'] == summary['champion']['trajectory']):
            best_plot = reviewed_rows
    if not best_plot:
        raise ValueError('Champion has no audited trajectory.')
    rows, curve = best_plot
    fig, (path, metric) = plt.subplots(1, 2, figsize=(12, 5))
    path.plot([float(r['x']) for r in rows], [float(r['z']) for r in rows], color='#2563eb')
    path.scatter(float(rows[0]['x']), float(rows[0]['z']), marker='o', color='#16803c', label='Start')
    path.scatter(float(rows[-1]['x']), float(rows[-1]['z']), marker='x', color='#dc2626', label='Last sample')
    path.set(xlabel='World X', ylabel='World Z', title='Observed top-down trajectory')
    path.axis('equal')
    path.legend()
    metric.plot(curve, color='#7c3aed')
    metric.set(xlabel='Decision', ylabel='Closest XYZ distance reduction (%)', title='Alive-grounded fitness')
    for ax in (path, metric):
        ax.grid(alpha=.2)
    fig.suptitle('Genetic champion: geometric distance is not native mission progress')
    fig.tight_layout()
    args.figure.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.figure, dpi=150)
    plt.close(fig)
    report = {'recorded_utc': datetime.now(timezone.utc).isoformat(), 'audit_passed': True,
              'run': str(args.run), 'fitness_mode': summary['fitness_mode'],
              'real_decisions': summary['real_decisions'], 'completed_trials': summary['completed_trials'],
              'partial_trials': summary['partial_trials'], 'generations': summary['generations_completed'],
              'stars': summary['full_start_stars'], 'deaths': summary['deaths'],
              'health_lost': summary['nonfatal_hp_lost'], 'action_counts': summary['action_counts'],
              'elapsed_seconds': summary['elapsed_seconds'], 'source_unchanged': True,
              'frozen_network_verified': True, 'champion_sha256': summary['champion_sha256'],
              'initial_head_ancestry': ancestry, 'reviewed_trajectories': reviewed,
              'missing_partial_trajectories': missing_partial, 'champion': summary['champion'],
              'confirmations': summary.get('confirmation_results', []),
              'outcomes': dict(Counter(r['outcome'] for r in records)), 'figure': str(args.figure)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k: report[k] for k in ('audit_passed', 'real_decisions', 'completed_trials',
                                           'partial_trials', 'generations', 'stars', 'deaths')}))


if __name__ == '__main__':
    main()
