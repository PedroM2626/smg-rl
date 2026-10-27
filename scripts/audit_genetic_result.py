"""Read-only audit and comparative figure for the completed one-hour genetic pilot."""

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from smg_rl.checkpoints import digest
from smg_rl.diagnostics import atomic_json
from smg_rl.dqn_trial import alive, locate_run, read
from smg_rl.genetic import genome_digest, next_population
from smg_rl.genetic_audit import audited_trajectory

OUTPUT = Path('docs/assets/performance/genetic-hour-result-20261026.json')
FIGURE = OUTPUT.with_name('genetic-hour-comparison-20261026.png')


def audit():
    supervisor = read(Path('.local/genetic-hour-state.json'))
    if supervisor['phase'] != 'completed':
        raise ValueError('Only the completed, reviewed pilot is reportable as completed.')
    run = locate_run(supervisor['genetic'])
    final = read(run / 'summary.json')
    if final['state'] != 'completed' or final['error'] or alive(read(run / 'controller.json')):
        raise ValueError('Genetic controller is not cleanly stopped.')
    source = Path(final['source'])
    if any(digest(source / name) != sha for name, sha in final['source_hashes'].items()):
        raise ValueError('The immutable source checkpoint changed.')
    if digest(run / 'champion.pt') != final['champion_sha256']:
        raise ValueError('Champion checkpoint hash failed.')
    champion = torch.load(run / 'champion.pt', map_location='cpu', weights_only=True)
    original = torch.load(source / 'agent.pt', map_location='cpu', weights_only=True)
    if (champion['source_online'].keys() != original['online'].keys()
            or not all(torch.equal(v, original['online'][k]) for k, v in champion['source_online'].items())
            or genome_digest(champion['genome'].numpy()) != final['champion']['genome_sha256']):
        raise ValueError('Frozen source weights or evolved genome identity failed.')
    trials = [json.loads(line) for line in (run / 'trials.jsonl').read_text().splitlines()]
    if (sum(t['decisions'] for t in trials) != final['real_decisions']
            or sum(t['fitness_eligible'] for t in trials) != final['completed_trials']
            or sum(not t['fitness_eligible'] for t in trials) != final['partial_trials']
            or sum(t['ram_star'] and t['fitness_eligible'] for t in trials) != final['full_start_stars']):
        raise ValueError('Final attempt/decision/completion accounting failed.')
    config = read(run / 'configuration.json')
    bounds = read(Path(config['paths']['calibration']))['variations']['sweet_sweet']
    for trial in trials:
        audited_trajectory(trial, bounds, config['environment']['max_step_distance'])
    previous = None
    for index in range(final['generations_completed']):
        meta = read(run / f'generation-{index:04d}.json')
        with np.load(run / f'generation-{index:04d}.npz', allow_pickle=False) as archive:
            population, fitness = archive['genomes'].copy(), archive['fitness'].copy()
        records = {t['individual']: t for t in trials if t['phase'] == 'evolution' and
                   t['generation'] == index and t['fitness_eligible']}
        if len(records) != final['population'] or not meta['review_passed'] or not np.isfinite(fitness).all():
            raise ValueError('An incomplete generation was reproduced.')
        for i, genome in enumerate(population):
            if (genome_digest(genome) != records[i]['genome_sha256']
                    or fitness[i] != records[i]['fitness_world_units']):
                raise ValueError('Generation archive and physical trial identity differ.')
        if previous is not None and not np.array_equal(population, previous):
            raise ValueError('Archived reproduction does not match the retained RNG state and algorithm.')
        rng = np.random.default_rng()
        rng.bit_generator.state = meta['rng_state_before_reproduction']
        previous = next_population(population, fitness, rng, sigma=final['mutation_std'])
    completed = [t for t in trials if t['phase'] == 'evolution' and t['fitness_eligible']]
    if max(t['fitness_world_units'] for t in completed) != final['champion']['fitness_world_units']:
        raise ValueError('The champion was not selected by maximum eligible physical fitness.')
    comparison = {}
    fig, (top, curve) = plt.subplots(1, 2, figsize=(12, 6), constrained_layout=True)
    for label, title, color in [('source_baseline', 'Original IQN head', '#d97706'),
                                ('genetic_champion', 'Evolved action head', '#2563eb')]:
        records = [t for t in trials if t['phase'] == 'frozen_confirmation' and t['individual'] == label]
        if sorted(t['seed'] for t in records) != [23, 47]:
            raise ValueError('Frozen canonical confirmation is incomplete.')
        comparison[label] = records
        rows, advances = audited_trajectory(records[0], bounds, config['environment']['max_step_distance'])
        points = np.array([[float(row[k]) for k in ('x', 'z')] for row in rows])
        top.plot(*points.T, color=color, linewidth=1.6, alpha=.85, label=title)
        best = int(np.argmax(advances))
        top.scatter(*points[best], color=color, marker='D', s=55, zorder=5)
        curve.plot(advances, color=color, label=f'{title}: {max(advances):.2f}%')
    top.set(xlabel='World X', ylabel='World Z', title='Top-down recorded trajectories')
    top.set_aspect('equal', adjustable='datalim')
    top.scatter(bounds['start'][0], bounds['start'][2], color='#059669', marker='o', label='Canonical start')
    curve.set(xlabel='Decision step', ylabel='Grounded high-water projection (%)',
              title='Fitness toward the calibrated near-star position', ylim=(0, 10))
    for axis in (top, curve):
        axis.grid(alpha=.2)
        axis.legend(fontsize=8)
    fig.suptitle('Sweet Sweet: frozen source versus genetic head — zero RAM-confirmed stars')
    fig.supxlabel('Diamonds: peak fitness. Projection is not native mission progress. Same savestate; deterministic confirmations.', fontsize=9)
    fig.savefig(FIGURE, dpi=150)
    plt.close(fig)
    overnight = read(Path('.local/smg1-overnight-state.json'))
    job = overnight['games']['smg1']['jobs'][-1]
    iqn_run = locate_run(job)
    controller, status = read(iqn_run / 'controller.json'), read(iqn_run / 'status.json')
    if (overnight['phase'] != 'running' or not alive(overnight['supervisor']) or not alive(controller)
            or Path(controller['resume']).resolve() != source.resolve()
            or status['source_count'] != final['source_real_decisions'] or status['error']):
        raise ValueError('Actual IQN resumption identity/health failed.')
    telemetry = read(Path('.local/impala-telemetry-state.json'))
    evidence = {'recorded_utc': datetime.now(timezone.utc).isoformat(), 'run': str(run),
                'audit_passed': True, 'all_trial_csv_fitness_recomputed': len(trials),
                'generation_genomes_fitness_and_rng_lineage_verified': final['generations_completed'],
                'source_checkpoint_unchanged': True, 'frozen_source_tensors_equal': True,
                'gradient_updates': final['gradient_updates'], 'elapsed_seconds': final['elapsed_seconds'],
                'real_decisions': final['real_decisions'], 'decisions_by_phase': dict(Counter({
                    phase: sum(t['decisions'] for t in trials if t['phase'] == phase)
                    for phase in {t['phase'] for t in trials}})),
                'evolution_completed_trials': len(completed), 'confirmation_completed_trials': 4,
                'partial_trials': final['partial_trials'], 'partial_decisions': sum(t['decisions'] for t in trials if not t['fitness_eligible']),
                'generations_completed': final['generations_completed'], 'full_start_stars': final['full_start_stars'],
                'champion': final['champion'], 'champion_sha256': final['champion_sha256'],
                'confirmation': comparison, 'improvement_ratio': final['champion']['fitness_world_units'] /
                    comparison['source_baseline'][0]['fitness_world_units'],
                'interpretation': 'Local improvement of a transferred action head under a different fitness/horizon; not a matched GA-versus-IQN training comparison. Same restored-state confirmations are not unseen tests. Geometric projection and mapped-opening fractions are not native mission progress.',
                'iqn_resumed_run': str(iqn_run), 'iqn_real_at_review': status['real_decisions'],
                'iqn_optimizer_updates_at_review': status['optimizer_updates'],
                'iqn_source_real': status['source_count'], 'iqn_deadline_utc': status['deadline_utc'],
                'telemetry_alive': alive(telemetry['controller']), 'figure': str(FIGURE)}
    if not atomic_json(OUTPUT, evidence):
        raise OSError('Could not persist the final genetic audit.')
    print(json.dumps({'audit_passed': True, 'output': str(OUTPUT), 'figure': str(FIGURE),
                      'improvement_ratio': evidence['improvement_ratio']}))


if __name__ == '__main__':
    audit()
