"""Audit compact full-visual genetic lineage and report physical outcomes only."""

import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import psutil
import torch

from smg_rl.checkpoints import digest
from smg_rl.config import load_config
from smg_rl.diagnostics import atomic_json
from smg_rl.dqn import DistributionalAgent
from smg_rl.dqn_trial import alive, locate_run, read
from smg_rl.genetic import genome_digest, next_population
from smg_rl.genetic_audit import audited_trajectory, counts
from smg_rl.genetic_visual import GeneticVisualPolicy
from smg_rl.replay import replay_signature

OUTPUT = Path('docs/assets/performance/genetic-visual-result-20261027.json')
FIGURE = OUTPUT.with_suffix('.png')


def visual_trial_summary(trials, baseline_hash):
    eligible = [t for t in trials if t['fitness_eligible']]
    mutated = [t for t in eligible if t.get('visual_genome_delta_l2', 0) > 0]
    if not mutated or not eligible:
        raise ValueError('No eligible physical trials with mutated vision to compare.')
    return {'completed_mutated_visual_trials': len(mutated),
            'completed_baseline_trials': sum(t['genome_sha256'] == baseline_hash for t in eligible),
            'best_complete': max(eligible, key=lambda t: t['fitness_world_units']),
            'best_mutated_visual': max(mutated, key=lambda t: t['fitness_world_units'])}


def audit():
    state = read(Path('.local/genetic-visual-20261027-state.json'))
    run = locate_run(state['genetic'])
    final = read(run / 'summary.json')
    if (state['phase'] != 'completed' or final['state'] != 'completed' or final['error']
            or final['policy_mode'] != 'full_visual' or final['stop_reason'] != 'scheduled_deadline'
            or final['gradient_updates'] != 0 or alive(state['supervisor']) or alive(read(run / 'controller.json'))):
        raise ValueError('Full-visual controller has not cleanly completed its finite allocation.')
    emulators = [p.pid for p in psutil.process_iter(['name']) if (p.info['name'] or '').lower() == 'dolphin.exe']
    if emulators:
        raise ValueError('Emulator processes remain; verify ownership before reporting shutdown.')
    source = Path(final['source'])
    if any(digest(source / name) != sha for name, sha in final['source_hashes'].items()):
        raise ValueError('The original source checkpoint changed.')
    torch.set_num_threads(1)
    config = load_config('config/genetic-smg1.yaml')
    agent = DistributionalAgent.load_policy(source, replay_signature(config), 'cpu')
    policy = GeneticVisualPolicy(agent)
    if policy.layout != read(run / 'gene-layout.json') or policy.layout_sha256 != final['gene_layout_sha256']:
        raise ValueError('Recorded visual genotype layout/scales differ from the actual policy.')
    baseline = np.load(run / 'baseline-genome.npy', allow_pickle=False)
    if genome_digest(baseline) != final['baseline_genome_sha256']:
        raise ValueError('The source baseline genome changed.')
    policy.source = baseline.copy()
    champion = torch.load(run / 'champion.pt', map_location='cpu', weights_only=True)
    if (digest(run / 'champion.pt') != final['champion_sha256'] or champion['policy_mode'] != 'full_visual'
            or champion['gene_layout'] != policy.layout
            or genome_digest(champion['genome'].numpy()) != final['champion']['genome_sha256']
            or not all(torch.equal(v, champion['source_online'][k]) for k, v in agent.online.state_dict().items())):
        raise ValueError('Final champion/source tensor identity failed.')
    trials = [json.loads(line) for line in (run / 'trials.jsonl').read_text().splitlines()]
    totals = counts(trials)
    if (any(totals[k] != final[k] for k in ('real_decisions', 'completed_trials', 'partial_trials', 'full_start_stars'))
            or sum(final['action_counts']) != final['real_decisions']
            or any(t['start_kind'] != 'full_start' or t['phase'] != 'evolution' for t in trials)):
        raise ValueError('Physical attempt/action/provenance accounting failed.')
    bounds = read(Path(config['paths']['calibration']))['variations']['sweet_sweet']
    last_time = 0.
    grouped = defaultdict(dict)
    for trial in trials:
        rows, _ = audited_trajectory(trial, bounds, config['environment']['max_step_distance'])
        if rows:
            last_time = max(last_time, float(rows[-1]['time']))
        key = (trial['generation'], trial['individual'])
        if trial['individual'] in grouped[trial['generation']]:
            raise ValueError(f'Duplicate trial identity: {key}')
        grouped[trial['generation']][trial['individual']] = trial
    initial = read(run / 'population-initial.json')
    if digest(run / 'population-initial.npz') != initial['sha256']:
        raise ValueError('Initial full population integrity failed.')
    with np.load(run / 'population-initial.npz', allow_pickle=False) as archive:
        population = archive['genomes'].copy()
    rng = np.random.default_rng()
    rng.bit_generator.state = initial['rng_state']
    trend = []
    for generation in range(final['generations_completed'] + 1):
        records = grouped[generation]
        if not np.isfinite(population).all() or population.shape != (24, final['genome_parameters']):
            raise ValueError('Reconstructed visual population dimensions/finite values failed.')
        for i, record in records.items():
            if genome_digest(population[i]) != record['genome_sha256']:
                raise ValueError('Candidate genome differs from reconstructed genotype lineage.')
            delta = policy.visual_delta_l2(population[i])
            if not np.isclose(delta, record['visual_genome_delta_l2'], rtol=1e-10, atol=1e-10):
                raise ValueError('Reported visual evolution differs from reconstructed CNN/projection genes.')
        if generation == final['generations_completed']:
            break  # Final partial generation is never used to reproduce.
        meta = read(run / f'generation-{generation:04d}.json')
        with np.load(run / f'generation-{generation:04d}.npz', allow_pickle=False) as archive:
            fitness = archive['fitness'].copy()
            if 'genomes' in archive:
                raise ValueError('Unexpected full snapshots in the compact generation journal.')
        if (len(records) != 24 or not meta['review_passed'] or meta['rng_state_before_reproduction'] != rng.bit_generator.state
                or any(not records[i]['fitness_eligible'] or fitness[i] != records[i]['fitness_world_units']
                       or records[i]['genome_sha256'] != meta['trial_genome_hashes'][str(i)] for i in range(24))):
            raise ValueError('Completed generation physical scores or mutation RNG failed.')
        mutated_scores = [r['fitness_world_units'] for r in records.values() if r['visual_genome_delta_l2'] > 0]
        trend.append((generation, float(fitness.max()), max(mutated_scores), float(fitness.mean())))
        population = next_population(population, fitness, rng, mutation_scale=policy.mutation_scale)
        if generation % 15 == 0:
            print(json.dumps({'audited_generation': generation, 'complete_generations': final['generations_completed']}), flush=True)
    pending = read(run / 'population-final.json')
    if (digest(run / 'population-final.npz') != pending['sha256'] or pending['sha256'] != final['final_population_sha256']
            or pending['rng_state'] != rng.bit_generator.state or pending['generation'] != final['generations_completed']):
        raise ValueError('Final population snapshot/RNG integrity failed.')
    with np.load(run / 'population-final.npz', allow_pickle=False) as archive:
        if not np.array_equal(population, archive['genomes']):
            raise ValueError('Reconstructed pending population differs from final snapshot.')
    summary = visual_trial_summary(trials, final['baseline_genome_sha256'])
    baseline_repeats = dict(Counter(str(t['geometric_projection_percent']) for t in trials
        if t['fitness_eligible'] and t['genome_sha256'] == final['baseline_genome_sha256']))
    if summary['best_complete']['fitness_world_units'] != final['champion']['fitness_world_units']:
        raise ValueError('The champion does not match the best eligible physical fitness.')
    chosen = [final['champion'], summary['best_mutated_visual']]
    fig, (top, plot) = plt.subplots(1, 2, figsize=(12, 6), constrained_layout=True)
    for trial, color, label in zip(chosen, ['#2563eb', '#d97706'], ['Unchanged baseline', 'Best mutated vision']):
        rows, advances = audited_trajectory(trial, bounds, config['environment']['max_step_distance'])
        positions = np.array([[float(row[k]) for k in ('x', 'z')] for row in rows])
        top.plot(*positions.T, color=color, alpha=.85, label=f"{label}: {trial['geometric_projection_percent']:.2f}%")
        top.scatter(*positions[int(np.argmax(advances))], marker='D', s=50, color=color)
    top.set(xlabel='World X', ylabel='World Z', title='Recorded best trajectories (diamonds: peak fitness)')
    top.set_aspect('equal', adjustable='datalim')
    series = np.asarray(trend)
    length = np.linalg.norm(np.asarray(bounds['goal'])[[0, 2]] - np.asarray(bounds['start'])[[0, 2]])
    for column, label, color in [(1, 'Population best', '#2563eb'), (2, 'Best with mutated vision', '#d97706'),
                                  (3, 'Population mean', '#64748b')]:
        plot.plot(series[:, 0] + 1, 100 * series[:, column] / length, label=label, color=color, alpha=.85)
    plot.set(xlabel='Completed generation (1-based)', ylabel='Grounded goal projection (%)',
             title='Unlocked visual evolution did not beat the baseline', ylim=(0, 10))
    for axis in (top, plot):
        axis.grid(alpha=.2)
        axis.legend(fontsize=8)
    fig.suptitle('Sweet Sweet: full-visual genetic trial — zero RAM-confirmed stars')
    fig.supxlabel('Geometric projection is not native mission progress. No new frozen confirmations were run.', fontsize=9)
    fig.savefig(FIGURE, dpi=150)
    plt.close(fig)
    names = read(source / 'manifest.json')['signature']['action_names']
    evidence = {'recorded_utc': datetime.now(timezone.utc).isoformat(), 'audit_passed': True, 'run': str(run),
        'all_controllers_stopped': True, 'remaining_emulators': emulators, 'stop_reason': final['stop_reason'],
        'supervisor_completed_at': state['completed_at'], 'elapsed_seconds': final['elapsed_seconds'], 'counts': totals,
        'new_generations': final['generations_completed'], 'genome_parameters': final['genome_parameters'],
        'visual_evolvable_parameters': final['visual_evolvable_parameters'], 'gradient_updates': final['gradient_updates'],
        'no_prior_fitness_reused': final['inherited_completed_trials'] == 0, 'source_hashes_unchanged': True,
        'initial_final_population_hashes_and_rng_verified': True, 'all_physical_trial_fitness_recomputed': len(trials),
        'complete_generation_genotypes_and_mutation_rng_reconstructed': final['generations_completed'],
        'actual_visual_genome_deltas_recomputed': len(trials), 'summary': summary,
        'baseline_repeat_geometric_percent_counts': baseline_repeats,
        'champion_sha256': final['champion_sha256'], 'champion_equals_unchanged_baseline':
            final['champion']['genome_sha256'] == final['baseline_genome_sha256'],
        'best_mutated_relative_fitness': summary['best_mutated_visual']['fitness_world_units'] / final['champion']['fitness_world_units'],
        'deaths': final['deaths'], 'nonfatal_hp_lost': final['nonfatal_hp_lost'],
        'action_occupancy': {name: {'count': n, 'percent': 100 * n / final['real_decisions']} for name, n in zip(names, final['action_counts'])},
        'real_decisions_per_second': final['real_decisions'] / final['elapsed_seconds'], 'new_frozen_confirmation_trials': 0,
        'last_emulator_sample_utc': datetime.fromtimestamp(last_time, timezone.utc).isoformat(),
        'last_sample_minus_deadline_seconds': last_time - datetime.fromisoformat(final['deadline_utc']).timestamp(),
        'interpretation': 'Visual parameters were genuinely mutated and evaluated, but none surpassed the unchanged ancestral baseline within this allocation. This does not isolate a causal explanation: search dimension and mutation scaling also changed. Geometric projection/opening fractions are not native mission percentage. Collection maxima are not new frozen evaluation results.',
        'figure': str(FIGURE)}
    if not atomic_json(OUTPUT, evidence):
        raise OSError('Could not persist visual genetic results.')
    print(json.dumps({'audit_passed': True, 'counts': totals, 'summary': summary, 'figure': str(FIGURE)}), flush=True)


if __name__ == '__main__':
    audit()
