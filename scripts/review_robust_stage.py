"""Audit robust-genetic evidence without running a game or learning."""

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from smg_rl.checkpoints import digest
from smg_rl.genetic import genome_digest
from smg_rl.robust_genetic import HEAD, SOURCE, aggregate_draws, calibration_decision, draw_seed

if __package__:
    from .audit_genetic_distance import audit_trial
else:
    from audit_genetic_distance import audit_trial

ACTIONS = {'none': 0, 'd': 1, 'a': 2, 'w': 3, 's': 4, 'jump': 5, 'spin': 6, 'crouch': 7}


def review(run):
    run = Path(run)
    summary = json.loads((run / 'summary.json').read_text(encoding='utf-8'))
    config = json.loads((run / 'configuration.json').read_text(encoding='utf-8'))
    if (summary['state'] not in ('completed', 'blocked') or summary.get('error')
            or not summary['source_unchanged'] or not summary['initial_head_unchanged']
            or config['curriculum']['enabled'] or config['regional_reward']['enabled'] or 'route' in config['paths']):
        raise ValueError('Stage is not clean or does not have the declared route-free runtime.')
    if any(digest(SOURCE / key) != value for key, value in summary['source_hashes'].items()):
        raise ValueError('Protected source hashes changed.')
    if digest(HEAD) != summary['initial_head_sha256']:
        raise ValueError('Initial head hash changed.')
    if 'frozen_network_before' in summary and summary['frozen_network_before'] != summary.get('frozen_network_after'):
        raise ValueError('Frozen visual network changed.')
    records = [json.loads(line) for line in (run / 'trials.jsonl').read_text().splitlines()]
    if (sum(r['decisions'] for r in records) != summary['real_decisions']
            or sum(bool(r['fitness_eligible']) for r in records) != summary['completed_trials']
            or sum(not r['fitness_eligible'] for r in records) != summary['partial_trials']
            or sum(bool(r['ram_star']) for r in records if r['fitness_eligible']) != summary['full_start_stars']):
        raise ValueError('Stage ledger and counters do not reconcile.')
    bounds = json.loads(Path(config['paths']['calibration']).read_text(encoding='utf-8'))['variations']['sweet_sweet']
    verified = missing_partial = 0
    for record in records:
        audited = audit_trial(record, bounds, config['environment']['max_step_distance'])
        if audited is None:
            missing_partial += 1
            continue
        rows, _ = audited
        action_bytes = bytes(ACTIONS[row['action']] for row in rows[1:])
        if (hashlib.sha256(action_bytes).hexdigest() != record['action_sha256']
                or hashlib.sha256(action_bytes[:60]).hexdigest() != record['action_prefix60_sha256']):
            raise ValueError('Action hash and recorded applied actions differ.')
        verified += 1
    stage = summary['stage']
    if stage == 'calibration':
        eligible = [r for r in records if r['fitness_eligible']]
        baseline = next(r for r in eligible if r['sigma'] == 0)
        decision = calibration_decision(eligible, baseline)
        if decision != summary['decision']:
            raise ValueError('Recomputed calibration decision differs.')
    elif stage == 'sampling-smoke':
        if summary['state'] != 'completed' or not summary['parity_passed']:
            raise ValueError('Frozen argmax parity was not established.')
        stochastic = [r for r in records if r.get('evaluation_mode') == 'epsilon' and r['fitness_eligible']]
        scores = aggregate_draws([(r, None) for r in stochastic], 1, 3)
        if not math.isclose(float(scores[0]), summary['epsilon_mean'], abs_tol=1e-8):
            raise ValueError('Sampling-smoke epsilon mean differs from physical draws.')
        decision = {'parity_passed': True, 'epsilon_mean': float(scores[0])}
    elif stage.startswith('arm-'):
        protocol = summary['protocol']
        for generation in range(summary['generations_completed']):
            meta = json.loads((run / f'generation-{generation:04d}.json').read_text())
            with np.load(run / f'generation-{generation:04d}.npz', allow_pickle=False) as data:
                genomes, scores, greedy = data['genomes'], data['selection_fitness'], data['greedy_fitness']
            if [genome_digest(x) for x in genomes] != meta['genome_hashes']:
                raise ValueError('Generation genome hashes differ.')
            selected = [r for r in records if r['generation'] == generation and r['phase'] == 'selection']
            for row in selected:
                if row['genome_sha256'] != meta['genome_hashes'][row['individual']]:
                    raise ValueError('Selection draw belongs to a different genotype.')
                if row['action_seed'] != draw_seed(generation, row['draw'], namespace=11):
                    raise ValueError('Training draw seed pairing failed.')
            calculated = aggregate_draws([(r, None) for r in selected], 48, protocol['selection_draws'])
            diagnostics = selected if protocol['arm'] == 'C' else [r for r in records
                            if r['generation'] == generation and r['phase'] == 'argmax_diagnostic']
            calculated_greedy = aggregate_draws([(r, None) for r in diagnostics], 48, 1)
            if not np.allclose(scores, calculated, atol=1e-8, rtol=0) or not np.allclose(greedy, calculated_greedy, atol=1e-8, rtol=0):
                raise ValueError('Generation aggregated score differs from individual draws.')
            if meta['top8'] != np.argsort(-scores, kind='stable')[:8].tolist():
                raise ValueError('Generation selection ordering differs.')
        decision = {'generations_verified': summary['generations_completed']}
    elif stage == 'confirmation':
        if summary['state'] != 'completed':
            raise ValueError('Incomplete confirmation cannot establish a result.')
        entries = summary['candidates']
        for i, entry in enumerate(entries):
            samples = [r for r in records if r['phase'] == 'confirmation' and r['individual'] == i]
            if len(samples) != 5 or any(r['action_seed'] != draw_seed(0, r['draw'], namespace=99) for r in samples):
                raise ValueError('Held-out action seeds or counts are invalid.')
            if not math.isclose(entry['epsilon_mean'], np.mean([r['fitness_world_units'] for r in samples]), abs_tol=1e-8):
                raise ValueError('Held-out mean does not match physical draws.')
        decision = {'candidates_verified': len(entries), 'independent_final_winner_test': False}
    else:
        raise ValueError('Unknown stage.')
    return {'audit_passed': True, 'stage': stage, 'state': summary['state'], 'run': str(run),
            'summary_sha256': digest(run / 'summary.json'), 'trials_sha256': digest(run / 'trials.jsonl'),
            'source_unchanged': True, 'network_digest_recorded': 'frozen_network_before' in summary,
            'real_decisions': summary['real_decisions'], 'completed_trials': summary['completed_trials'],
            'partial_trials': summary['partial_trials'], 'stars': summary['full_start_stars'],
            'deaths': summary['deaths'], 'health_lost': summary['nonfatal_hp_lost'],
            'elapsed_seconds': summary['elapsed_seconds'], 'verified_trajectories': verified,
            'missing_partial_trajectories': missing_partial, 'decision': decision}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Refusing to overwrite existing evidence.')
    result = review(args.run)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
